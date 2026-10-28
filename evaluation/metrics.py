import sacrebleu

def compute_bleu(hypotheses: list, references: list) -> float:
    """
    Computes corpus-level BLEU score using sacrebleu.
    references is a list of reference strings (or list of lists).
    """
    if not hypotheses or not references:
        return 0.0
    # sacrebleu expects references as a list of lists of strings: [[ref1, ref2, ...]]
    if isinstance(references[0], str):
        ref_list = [[r.strip() for r in references]]
    else:
        ref_list = references
    hyp_list = [h.strip() if h.strip() else "<empty>" for h in hypotheses]
    
    bleu = sacrebleu.corpus_bleu(hyp_list, ref_list)
    return round(float(bleu.score), 2)

def compute_chrf(hypotheses: list, references: list) -> float:
    """
    Computes corpus-level chrF score using sacrebleu.
    Particularly informative for morphologically rich Indian languages like Hindi.
    """
    if not hypotheses or not references:
        return 0.0
    if isinstance(references[0], str):
        ref_list = [[r.strip() for r in references]]
    else:
        ref_list = references
    hyp_list = [h.strip() if h.strip() else "<empty>" for h in hypotheses]
    
    chrf = sacrebleu.corpus_chrf(hyp_list, ref_list)
    return round(float(chrf.score), 2)

def compute_sentence_metrics(hyp: str, ref: str) -> dict:
    """
    Computes sentence-level BLEU and chrF.
    """
    hyp = hyp.strip() if hyp.strip() else "<empty>"
    ref = ref.strip()
    bleu = sacrebleu.sentence_bleu(hyp, [ref]).score
    chrf = sacrebleu.sentence_chrf(hyp, [ref]).score
    return {
        "bleu": round(float(bleu), 2),
        "chrf": round(float(chrf), 2)
    }
