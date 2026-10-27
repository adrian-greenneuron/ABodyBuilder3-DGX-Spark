"""Explicit rescue dispatch; successful R8 preparation is returned unchanged."""
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
    policy=options.rescue_policy
    if policy=='off':return original(raw,options)
    if policy not in ('local','restrained'):raise ValueError('Unknown explicit heavy-atom rescue policy.')
    options.validate()
    started=time.perf_counter()
    try:
        value=original(raw,options)
        value[2]['rescue']={'policy':policy,'applied':False}
        return value
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
            if getattr(exc,'report',None) is not None:
                audit['rescue']['rescue_failure_audit']=exc.report
            raise error_type(str(exc),audit) from exc
        report['rescue'].update(applied=True,policy=policy,trigger_failure=rejected.report,
                               trigger_seconds=rejected.report.get('seconds'))
        report['seconds']=time.perf_counter()-started
        return prepared,relaxed,report
