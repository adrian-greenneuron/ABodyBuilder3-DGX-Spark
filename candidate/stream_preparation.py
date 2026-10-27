"""Bounded JSON-lines IPC to a lightweight, process-isolated chemistry pool.

Model workers never fork. Spawned chemistry workers import this small module,
not the neural CLI. Credits bound accepted/incomplete work to twice the worker
count; writer backpressure propagates through the existing inference pipeline.
"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
import json
import multiprocessing
import queue
import subprocess
import sys
import threading
from pathlib import Path


class PreparationStream:
    def __init__(self, options, workers):
        if not 1 <= workers <= 20 or options.relax != "none":
            raise ValueError("Streaming preparation requires 1–20 workers and no relaxation.")
        self.closed=False;self.results=[];self.error=None;self.seen=set()
        self.limit=2*workers;self.credits=threading.Semaphore(self.limit)
        self.lock=threading.Lock();self.max_inflight=0;self.inflight=0
        self.process=subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve()),"--controller"],
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        self.process.stdin.write(json.dumps({"workers":workers,"options":asdict(options)})+"\n")
        self.process.stdin.flush()
        self.reader=threading.Thread(target=self._read,daemon=True);self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                row=json.loads(line)
                with self.lock:self.results.append(row);self.inflight-=1
                self.credits.release()
        except Exception as exc:
            self.error=f"Streaming controller output failed: {exc}"
        finally:
            if not self.closed:self.error=self.error or "Streaming controller exited before input closed."
            # Wake any blocked writers so they can observe the failure.
            for _ in range(self.limit):self.credits.release()

    def submit(self,path):
        name=str(Path(path).resolve());self.credits.acquire()
        with self.lock:
            if self.error:raise RuntimeError(self.error)
            if self.closed:raise RuntimeError("Streaming preparation is closed.")
            if name in self.seen:
                self.credits.release();raise ValueError("Duplicate preparation path.")
            self.seen.add(name);self.inflight+=1;self.max_inflight=max(self.max_inflight,self.inflight)
            try:
                self.process.stdin.write(json.dumps({"path":name})+"\n");self.process.stdin.flush()
            except Exception:
                self.inflight-=1;self.credits.release();raise

    def finish(self,raise_errors=True):
        if not self.closed:
            self.closed=True
            try:self.process.stdin.close()
            except BrokenPipeError:pass
            code=self.process.wait();self.reader.join()
            self.process.stdout.close()
            if code:self.error=self.error or f"Streaming controller exited with {code}."
        failed=[x for x in self.results if x["status"]=="failed"]
        if len(self.results)!=len(self.seen):self.error=self.error or "Streaming preparation lost a result."
        if raise_errors and (self.error or failed):
            raise RuntimeError(self.error or f"Preparation failed for {len(failed)} structures; raw files/audits retained.")
        return {"accepted":len(self.seen),"completed":len(self.results),"max_inflight":self.max_inflight,
                "inflight_limit":self.limit,"workers":len({x.get("worker_pid") for x in self.results}),
                "error":self.error,"results":self.results}


def _job(job):
    import os
    from preparation_pool import _prepare_one
    row=_prepare_one(job);row["worker_pid"]=os.getpid()
    return row


def controller():
    from preparation_pool import _initialise
    payload=json.loads(sys.stdin.readline());workers=payload["workers"];options=payload["options"]
    if not 1<=workers<=20 or options["relax"]!="none":raise ValueError("Invalid stream configuration.")
    incoming=queue.Queue(maxsize=2*workers);errors=[];sentinel=object()
    def read():
        try:
            for line in sys.stdin:incoming.put(json.loads(line))
        except Exception as exc:errors.append(repr(exc))
        finally:incoming.put(sentinel)
    reader=threading.Thread(target=read,daemon=True);reader.start()
    eof=False;pending={};seen=set()
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),initializer=_initialise) as pool:
        while not eof or pending:
            while len(pending)<2*workers and not eof:
                try:item=incoming.get(timeout=.02 if not pending else 0)
                except queue.Empty:break
                if item is sentinel:eof=True;break
                path=item["path"]
                if path in seen:raise ValueError("Duplicate stream path.")
                seen.add(path);pending[pool.submit(_job,(path,options))]=path
            if pending:
                completed,_=wait(pending,timeout=.02,return_when=FIRST_COMPLETED)
                for future in completed:
                    path=pending.pop(future)
                    try:row=future.result()
                    except Exception as exc:row={"path":path,"status":"failed","error":repr(exc),"warnings":[]}
                    print(json.dumps(row),flush=True)
        reader.join()
    if errors:raise RuntimeError(errors)


if __name__=="__main__":controller()
