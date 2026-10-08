This is a companion to [the main post](/). It covers the places where my Hangul handwriting model departs from the architecture in Alex Graves' 2013 paper, ["Generating Sequences With Recurrent Neural Networks"](https://arxiv.org/abs/1308.0850).

## Text embeddings

The model needs some way to know what to write. To do this, we convert the input string into a set of text embeddings. The model uses a sliding-window attention mechanism across these text embeddings as it writes.

Graves used non-learnable, one-hot embeddings for 57 different characters in his model. If I took this approach with Hangul, I would need over 11,000 embeddings. This wasn't going to work.

For my model, I borrowed some concepts from Unicode. Unicode decomposes each Korean character into a leading consonant (19), a vowel (21), and a trailing consonant (28). Each one of these components is called a jamo. I use a learnable rank-3 embedding for each jamo. To get my embedding for each character, I simply concatenate the embeddings for the leading consonant, the vowel, and the trailing consonant together to get one rank-9 embedding for each Korean character.

## Pressure

Graves' initial model predicted the x, y direction of the pen stroke. Initially, I did this too. However, I noticed that when I also trained on the pen pressure, the model was able to learn much faster on a much smaller amount of data.

This makes sense. You would expect that pen pressure would be a good indicator of when a stroke was almost finished or if it was still mid-stroke.

## Stopping mechanism

One of the trickiest things about this was teaching the model when it's time to stop generating strokes. The training data is rich with signals of what to draw next. But unfortunately, signals for when to stop drawing are much less frequent.

Graves' original model had a simple way to signal stopping. When the attention marker slides past the end of the input text, then stop generating strokes. This is elegant, but unfortunately it didn't work well on Hangul. The model would frequently stop too early, or it would stop too late, leaving a trail of garbage after the last stroke.

I borrowed the stopping mechanism from Google's [Sketch-RNN](https://arxiv.org/abs/1704.03477) (2017). In this mechanism, we add a special end-of-text token to the input text, and augment the training strokes with a tail of "parked-pen" strokes, so that the model learns how to behave when handwriting is coming to an end.

## Positional feedback

One early problem I ran into was that the model would lose track of where it was. The text would tend to drift too far up or down as it wrote from left to right.

To fix this, I modified the RNN loop to include the previous x, y position of the pen during each forward pass. This allows the model to generally write in a straight line.

## LSTM → GRU

Graves' original model used an LSTM (Long Short-Term Memory) architecture. LSTMs were introduced in 1997, and at the time of Graves' paper, were the state of the art for recurrent neural networks.

A year after Graves' paper, the GRU (Gated Recurrent Unit) was invented as a simpler alternative to an LSTM.

I chose to use a GRU for this project, for no particular reason other than to see if it would work, and it did!
