from core.vitals.cardiac_workload import CardiacWorkloadEngine, CardiacWorkloadResult

def compute_cardiac_workload(hr_bpm: float = 72.0, systolic_bp: float = 120.0) -> CardiacWorkloadResult:
    engine = CardiacWorkloadEngine()
    return engine.compute_workload(hr_bpm=hr_bpm, systolic_bp=systolic_bp)

__all__ = ['CardiacWorkloadEngine', 'CardiacWorkloadResult', 'compute_cardiac_workload']
