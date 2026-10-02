"""In-memory CredSweeper adapter. No files containing candidate values, no export."""
from contextlib import contextmanager
from importlib.metadata import version
import logging
from pathlib import PurePosixPath
from threading import RLock
from unittest.mock import patch
from privai_model.data.adapter import adapt_records

_LOG_LOCK = RLock()


@contextmanager
def private_scanner_logs():
    # CredSweeper debug messages can contain raw lines. Scope filters to its loggers,
    # restore them afterwards, and serialize our adapter's use of the scanner.
    class Drop(logging.Filter):
        def filter(self, record): return False
    filt=Drop()
    with _LOG_LOCK:
        loggers=[logging.getLogger('credsweeper')]
        loggers += [logging.getLogger(n) for n in list(logging.Logger.manager.loggerDict) if n.startswith('credsweeper.')]
        for logger in loggers: logger.addFilter(filt)
        try: yield
        finally:
            for logger in loggers: logger.removeFilter(filt)


def covers_span(line_data, offsets, target_start, target_end):
    """A finding must contain this candidate span, not merely occur on its line."""
    number=getattr(line_data,'line_num',None)
    start,end=getattr(line_data,'value_start',-1),getattr(line_data,'value_end',-1)
    if type(number) is not int or not 1<=number<=len(offsets) or start<0 or end<=start:
        return False
    offset=offsets[number-1]
    return offset+start<=target_start and target_end<=offset+end


class CredSweeperBaseline:
    def __init__(self, use_ml=True):
        try:
            from credsweeper import CredSweeper
            from credsweeper.common.constants import ThresholdPreset
        except ImportError:
            raise RuntimeError('Install baseline extra: uv sync --extra baselines --locked') from None
        self.use_ml=use_ml
        self.version=version('credsweeper')
        from credsweeper.credentials.credential_manager import CredentialManager
        class SerialCredentialManager(CredentialManager):
            def __init__(self):
                # Upstream 1.18.5 does list(Manager().list()), which is a plain list
                # after creating an unnecessary IPC server. This serial wrapper
                # uses the exact same storage type without a child process/socket.
                self.candidates=[]
        with private_scanner_logs(), patch('credsweeper.app.CredentialManager',SerialCredentialManager):
            self.scanner=CredSweeper(stdout=False,hashed=True,pool_count=1,depth=0,doc=False,
                                    ml_threshold=ThresholdPreset.medium if use_ml else 0,
                                    ml_threads_limit=2,ml_providers='CPUExecutionProvider')

    @property
    def metadata(self):
        return {'method':'CredSweeper','version':self.version,'output':'binary prediction',
                'use_ml':self.use_ml,'ml_threshold':'medium' if self.use_ml else 'disabled',
                'match_policy':'reported value span contains input candidate span',
                'score_note':'ML scores do not cover every rule or rejected candidate; not exposed as complete probabilities',
                'scope':'candidate-local supplied line/context, not whole repository scanning',
                'runtime_adapter':'serial plain-list CredentialManager, no IPC; upstream rules/filters/ML unchanged'}

    def predict(self, candidate_records):
        from credsweeper.file_handler.string_content_provider import StringContentProvider
        records=list(candidate_records)
        if any(not isinstance(r,dict) or r.get('granularity')!='value' for r in records):
            raise ValueError('baseline accepts value granularity only')
        examples=adapt_records(records,require_label=False)
        rows=[]
        for ex in examples:
            prefix=ex.before+'\n' if ex.before else ''
            text=prefix+ex.line+ ('\n'+ex.after if ex.after else '')
            lines=text.split('\n')
            offsets=[]; offset=0
            for line in lines:
                offsets.append(offset);offset+=len(line)+1
            suffix=PurePosixPath(str(ex.source.get('file_path') or 'candidate.env')).suffix
            suffix=suffix if suffix and suffix[1:].isalnum() else '.txt'
            provider=StringContentProvider(lines,file_path='candidate'+suffix,file_type=suffix)
            with private_scanner_logs():
                self.scanner.credential_manager.clear_credentials()
                try:
                    self.scanner.scan([provider],None)
                    self.scanner.post_processing(None)
                    found=any(covers_span(ld,offsets,len(prefix)+ex.start,len(prefix)+ex.end)
                              for candidate in self.scanner.credential_manager.get_credentials()
                              for ld in candidate.line_data_list)
                except Exception:
                    raise RuntimeError('CredSweeper scan failed; raw scanner error suppressed') from None
                finally:
                    self.scanner.credential_manager.clear_credentials()
            rows.append({'candidate_id':ex.candidate_id,'prediction':'MASK' if found else 'KEEP'})
        if len(rows)!=len(examples) or {r['candidate_id'] for r in rows}!={e.candidate_id for e in examples}:
            raise ValueError('baseline prediction coverage mismatch')
        return rows
