"""Experimental eager CUDA graph replay, with unchanged model operations.

Each graph owns its private allocation pool; pools are not shared between
graphs. Static input storage is shared because replay and CPU conversion are
strictly sequential in the single GPU-owning production pipeline.
"""
import time
import torch


class GraphForward:
    def __init__(self, forward, max_batch, max_length):
        self.forward = forward
        self.max_batch = max_batch
        self.max_length = max_length
        self.storage = None
        self.graphs = {}
        self.events = []

    def __call__(self, inputs, aatype, mask, **options):
        b,n=aatype.shape
        assert b<=self.max_batch and n<=self.max_length
        if self.storage is None:
            m,l=self.max_batch,self.max_length
            self.storage={
                'single':torch.empty(m*l*inputs['single'].shape[-1],device=aatype.device,dtype=inputs['single'].dtype),
                'pair':torch.empty(m*l*l*inputs['pair'].shape[-1],device=aatype.device,dtype=inputs['pair'].dtype),
                'aatype':torch.empty(m*l,device=aatype.device,dtype=aatype.dtype),
                'mask':torch.empty(m*l,device=mask.device,dtype=mask.dtype)}
        static={'single':self.storage['single'][:inputs['single'].numel()].view_as(inputs['single']),
                'pair':self.storage['pair'][:inputs['pair'].numel()].view_as(inputs['pair'])}
        static_type=self.storage['aatype'][:aatype.numel()].view_as(aatype)
        static_mask=self.storage['mask'][:mask.numel()].view_as(mask)
        for key in static:static[key].copy_(inputs[key])
        static_type.copy_(aatype);static_mask.copy_(mask)
        key=(b,n,tuple(sorted(options.items())))
        if key not in self.graphs:
            start=time.perf_counter()
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    warm=self.forward(static,static_type,static_mask,**options)
                    del warm
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                output=self.forward(static,static_type,static_mask,**options)
            self.graphs[key]=(graph,output)
            self.events.append({'batch_size':b,'padded_length':n,
                                'capture_and_warmup_seconds':time.perf_counter()-start})
        graph,output=self.graphs[key]
        graph.replay()
        return output
