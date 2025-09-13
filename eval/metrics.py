import editdistance

def cer(ref: str, hyp: str) -> float:
    return editdistance.eval(ref, hyp) / max(1, len(ref))