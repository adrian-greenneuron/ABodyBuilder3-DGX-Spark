"""Explicit rescue dispatch; successful R8 preparation is returned unchanged."""
import os
import time

FROZEN_REJECTION='Frozen heavy-atom geometry is incompatible with a planar, outward hydrogen and the angle-error limit; explicit heavy-atom repair is required.'


def is_frozen_rejection(error):
    seen=set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error,ValueError) and str(error)==FROZEN_REJECTION:return True
        error=error.__cause__
    return False


def dispatch(raw,options,original,error_type,runner=None):
    policy=os.environ.get('AB3_EXPERIMENTAL_RESCUE','off')
    if policy=='off':return original(raw,options)
    if policy not in ('local','restrained'):raise ValueError('Unknown explicit heavy-atom rescue policy.')
    if options.relax!='none' or options.ph not in (None,7.) or os.environ.get('AB3_EXPERIMENTAL_GEOMETRY')!='robust_planar':
        raise ValueError('The rescue experiment requires pH7, robust_planar H placement and no global relaxation option.')
    started=time.perf_counter()
    try:return original(raw,options)
    except error_type as rejected:
        if not is_frozen_rejection(rejected):raise
        if runner is None:
            from rejected_rescue import prepare_rejected
            runner=prepare_rejected
        try:
            prepared,relaxed,report=runner(raw,options,policy)
        except Exception as exc:
            audit=dict(rejected.report)
            audit.update(status='failed',error=str(exc),seconds=time.perf_counter()-started,
                rescue={'policy':policy,'applied':False,'trigger_failure':rejected.report,'rescue_failure':str(exc)})
            raise error_type(str(exc),audit) from exc
        report['rescue'].update(applied=True,policy=policy,trigger_failure=rejected.report,
                               trigger_seconds=rejected.report.get('seconds'))
        report['seconds']=time.perf_counter()-started
        return prepared,relaxed,report
