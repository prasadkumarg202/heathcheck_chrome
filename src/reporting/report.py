import time
import uuid
from typing import Dict, Any, Optional

def generate_health_report(
    patient_id: Optional[str] = None,
    session_id: Optional[str] = None,
    duration_s: float = 60.0,
    device_info: Optional[str] = None,
    vitals_data: Optional[Dict[str, Any]] = None,
    sqi_score: float = 85.0
) -> Dict[str, Any]:
    session_id = session_id or str(uuid.uuid4())[:8]
    patient_id = patient_id or 'ANON-' + str(uuid.uuid4())[:6]
    vitals = vitals_data or {}
    
    return {
        'report_id': 'REP-' + str(uuid.uuid4())[:8].upper(),
        'patient_id': patient_id,
        'session_id': session_id,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'duration_seconds': duration_s,
        'device': device_info or 'Standard Browser Camera / On-Device WebAssembly',
        'engine_provenance': {
            'engine_version': 'AuraPulse 2.0.0-prod',
            'algorithms': ['POS', 'CHROM', 'Multi-Zone-ROI', 'Dynamic-Skin-Mask'],
            'model_version': 'YuNet-2023Mar-ONNX',
            'execution_environment': '100% Client-Side Ephemeral RAM'
        },
        'signal_quality': {
            'overall_sqi': sqi_score,
            'rating': 'Excellent' if sqi_score >= 70 else ('Good' if sqi_score >= 50 else 'Poor')
        },
        'vitals': vitals,
        'disclaimer': 'Camera-based measurements are intended for wellness, research, or screening purposes. Not for medical diagnosis.'
    }
