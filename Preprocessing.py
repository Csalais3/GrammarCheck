import spacy
from spacy.language import Language
from spacy.tokens import Token, Doc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import errant

## Parameters
MODEL_CHECKPOINT     = "modelt"
BEAM_WIDTH           = 5
SPACY_MODEL          = "en_core_web_trf"
EXPLANATION_TEMPLATES = {
    "R:VERB:TENSE":   "Verb tense agreement: “{orig}” → “{cor}” to match the time frame.",
    "R:SVA":          "Subject-verb agreement: subject is singular, so verb must end in ‘-s’.",
}


## Preprocessing
nlp = spacy.load(SPACY_MODEL)

# Mark named-entity tokens to ignore
Token.set_extension("ignore_for_grammar", default=False, force=True)

@Language.component("mask_named_entities")
def mask_named_entities(doc: Doc) -> Doc:
    for ent in doc.ents:
        for token in ent:
            token._.ignore_for_grammar = True
    return doc

nlp.add_pipe("mask_named_entities", after="ner")

## Seq2Seq Correction 

hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
hf_model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

Doc.set_extension("corrected_text", default=None, force=True)

@Language.component("gec_correction")
def gec_correction(doc: Doc) -> Doc:
    inputs  = hf_tokenizer(doc.text, return_tensors="pt")
    outputs = hf_model.generate(**inputs, num_beams=BEAM_WIDTH)
    doc._.corrected_text = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return doc

nlp.add_pipe("gec_correction", last=True)

## ERRANT
annotator = errant.load("en")

def run_errant(orig: str, corr: str):
    """Diff original vs. corrected and return a list of edits."""
    annotation = annotator.annotate(orig, corr)
    return list(annotation)

## Explanation
Doc.set_extension("explanations", default=[], force=True)

def explain_edits(edits):
    expls = []
    for e in edits:
        tmpl = EXPLANATION_TEMPLATES.get(e.tag)
        if tmpl:
            expls.append(tmpl.format(orig=e.src, cor=e.tgt, subj=e.subj))
    return expls

## Ochestration

def process_sentence(text: str):
    doc = nlp(text)
    corrected    = doc._.corrected_text
    edits        = run_errant(text, corrected)
    explanations = explain_edits(edits)
    return {
        "original":     text,
        "corrected":    corrected,
        "explanations": explanations
    }
