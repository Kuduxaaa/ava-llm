# Georgian Language Modeling

## Overview

Ava is designed with Georgian language support as a primary goal. Georgian (ქართული) presents unique challenges for language modeling due to its rich morphology and unique script.

## Georgian Script

Georgian uses three scripts: Mkhedruli (მხედრული, modern), Asomtavruli, and Nuskhuri. Modern Georgian uses 33 letters in the Mkhedruli alphabet (Unicode range U+10D0–U+10FF).

Key characteristics:
- No uppercase/lowercase distinction in Mkhedruli
- Left-to-right writing direction
- Rich agglutinative morphology

## Tokenizer Recommendations

### BPE (Byte-Pair Encoding)
Recommended approach for Georgian:
- Train a custom BPE tokenizer on Georgian text corpus
- Vocabulary size: 32,000–50,000 tokens
- Include both Georgian and Latin characters for code-switching support

### SentencePiece
Alternative with good Georgian support:

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="georgian_corpus.txt",
    model_prefix="ava_georgian",
    vocab_size=32000,
    model_type="bpe",
    character_coverage=0.9995,
    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3,
)
```

### Tips
- Set `character_coverage=0.9995` to ensure all Georgian characters are covered
- Pre-normalize text: remove zero-width characters, normalize Unicode
- Consider adding morphological segmentation as a pre-tokenization step

## Morphological Considerations

Georgian is agglutinative with complex verb morphology:
- Verbs can have 7+ morphemes: preverb + version + person + root + thematic suffix + tense + number
- Example: გადმოგვიგზავნიდნენ (gadmogvigzavnidnen) = "they were apparently sending to us"

Implications for modeling:
- Larger vocabulary may help capture common morphological patterns
- Subword tokenization (BPE) naturally handles agglutination
- Mamba's sequential processing may handle Georgian word structure well since morpheme order is strictly sequential

## Dataset Sources

Potential Georgian text sources:
- Georgian Wikipedia
- Georgian news sites
- OASST multilingual dataset (includes Georgian conversations)
- Georgian National Corpus

## Training Tips for Georgian

1. **Tokenizer**: Train on a large Georgian corpus (>1GB of text)
2. **Mixed data**: Include some English data (5-10%) for better generalization
3. **Sequence length**: Georgian words can be long — use 512+ token sequences
4. **Evaluation**: Use perplexity on held-out Georgian text as primary metric
