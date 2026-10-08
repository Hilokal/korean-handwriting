The model behind [hangul.ink](/) is built on Alex Graves' 2013 paper, ["Generating Sequences With Recurrent Neural Networks"](https://arxiv.org/abs/1308.0850). That architecture was designed for English, and most of it carried over to Hangul untouched. This post is about the parts that didn't.

A quick sketch of the original, so the changes make sense. Graves' handwriting model is a stack of LSTM layers that reads the pen's previous movement and predicts the next one. The prediction isn't a single point but a mixture of Gaussians (a mixture density network, or MDN), which the sampler draws from. To know *what* to write, the network carries a soft attention window over the input text: a handful of Gaussian bumps whose centers only ever move rightward, so the window sweeps across the characters as the pen moves across the page. Everything below is a change to one of those pieces.

## Text embeddings: decompose the syllable

Graves conditioned his model on one-hot vectors, one slot per character, for an alphabet of 57 characters. Hangul has over 11,000 syllables in common use. A one-hot table that size would have been mostly untrained slots, since even 4,000 lines of handwriting touch only a fraction of them.

The fix is in how Hangul itself is built. Every syllable is a block assembled from two or three letters, called *jamo*: a leading consonant (19 possibilities), a vowel (21), and an optional trailing consonant (28). Unicode encodes syllables arithmetically from those three indices, so decomposing a character into its jamo is a few lines of integer math.

So instead of one embedding per syllable, the model learns three small tables, one per jamo position, and concatenates the three lookups into a single vector per syllable. I started with 3 dimensions per jamo and found that 8 worked noticeably better; beyond that nothing improved. A syllable the model has never seen still decomposes into jamo it has seen hundreds of times, which is what lets it generalize to new text.

Spaces and punctuation get their own slots in a separate table, so a comma's strokes attend to a real position in the text rather than bleeding into the neighbouring syllable.

## Pen pressure as a signal

Graves' pen data had only position. The Neo Smartpen also reports tip force, and I initially threw it away to keep the model faithful to the paper.

Adding it back, as both an input channel and a predicted output, made the model learn faster on less data. In hindsight that's not surprising: pressure is a strong hint about where you are within a stroke. It climbs at the start, holds through the middle, and tails off before the pen lifts. Giving the model that signal means it doesn't have to infer stroke phase from geometry alone. The demo renders this too, which is why the line width varies along each stroke.

## Positional feedback

Graves' model sees only *deltas*, the offset from the previous point. That makes it translation-invariant, which is nice for learning letter shapes, but it also means the model has no idea where it actually is on the page. Early versions would drift: the baseline would wander up or down across a line, and the errors compounded.

The change was small. At each step I feed the model the pen's absolute position (the running sum of all deltas so far) alongside the delta. Two extra numbers per timestep, and the lines come out straight.

## Knowing when to stop

This was the hardest problem, and the one I rewrote most often.

Graves stopped generating when the attention window slid past the last character. That's elegant, and it mostly worked for his English data. On Hangul it was unreliable in both directions: sometimes the window overshot and the model stopped mid-syllable, sometimes it lingered and the model kept producing a trail of garbage strokes after the real text had ended.

The underlying issue is that the training data is dense with examples of what to draw next and nearly empty of examples of *stopping*. Each line contains hundreds of points and exactly one ending.

I borrowed the solution from Google's [Sketch-RNN](https://arxiv.org/abs/1704.03477) (2017). Two things change. The text gets a special end-of-text token appended after the last real character, so the attention window has a trained position to settle on once the writing is done. And every training line gets a short tail of "parked pen" points appended after its last real stroke: zero movement, pen up, zero pressure. The model learns to produce that parked state when it reaches the end token, and generation stops when the window's peak reaches that token. It's a few extra points per line, but it turns stopping into something the model is actually trained to do.

## LSTM → GRU

Graves used LSTM cells, which were the state of the art for recurrent networks in 2013. The GRU (Gated Recurrent Unit) came out a year later as a simpler alternative with fewer gates and fewer parameters.

I used GRUs, for no better reason than curiosity about whether they'd work here. They did. The shipped model is three GRU layers of 128 units, with a 20-component mixture density head, and it fits in a 1.3 MB file that runs in your browser.

## Everything else

The MDN output head, the mixture-of-Gaussians attention window with ten bumps, the monotonic sweep across the text, the sampling with a bias knob for neater or messier handwriting: all of that is Graves' design and works on Hangul as published. If you want to understand those parts, Distill's ["Four Experiments in Handwriting with a Neural Network"](https://distill.pub/2016/handwriting/) is the best walkthrough I know of, and the [source code](https://github.com/Hilokal/korean-handwriting) for this model is available as a reference.
