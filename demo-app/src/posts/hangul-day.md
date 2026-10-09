Today is October 9, Happy Hangul Day!

If you didn't know, Hangul is the alphabet that is used for the written Korean language. The Hangul alphabet is unique, in that it was created relatively recently, in 1446, and was designed intentionally to be easy to learn.

The Koreans are rightfully very proud of this alphabet. And so October 9 is a national holiday to celebrate Hangul.

To celebrate Hangul Day this year, I am releasing an AI model that generates Hangul handwriting strokes. As far as I know, it's the first one that's publicly available.

You can try this model in your web browser. If you know Korean, you can type some Korean text and the model will generate a sequence of pen strokes to render that text. If you don't know Korean, you can use one of the pre-filled bubbles with some common phrases.

<!-- demo -->

Like most generative models, it includes a random sampling layer, so each generation from the model is unique.

The demo you see here is rendering strokes into an SVG animation, but there's no reason this couldn't also be set up to drive an auto-pen.

## Why?

I get this question a lot.

This project combined two of my interests: generative AI models, and the Korean language.

I have a long career in software development, but I find generative AI models fascinating. As a learning exercise, I wanted to try training my own generative model from scratch. I wanted to make mistakes, fix them, and learn in the process.

And I wanted to create something new. There are many publicly available English handwriting models, but this is the only Korean handwriting model.

### How is this different than other AI models?

There are many AI models these days, many of which are capable of creating images, even Hangul images.

The biggest difference is that this model is not generating pixels directly, but is instead generating a sequence of pen strokes. Each item in the sequence contains an x, y coordinate, a pen pressure, and a pen-up/pen-down instruction. This is why the output looks like an animation, and not a static image, and why the lines get thicker and thinner as the pen moves.

Also, this model is extremely small. It's about 350,000 parameters, which runs comfortably in your browser. In fact, there is no server.

## Architecture

This model's architecture is based on a 2013 paper called ["Generating Sequences With Recurrent Neural Networks"](https://arxiv.org/abs/1308.0850) by Alex Graves. In this paper, Graves makes the case that a recurrent neural network (or RNN) can be used to generate sequences of things. This seems obvious today, but at the time it was quite novel. In his paper, one of the example sequences he uses is handwriting, which is just a sequence of pen strokes.

A recurrent neural network is simply a model that runs in some kind of loop, and maintains some hidden state between each iteration of that loop. This is an old concept, but RNNs are still very relevant today.

Recent work on ["looped" transformers](https://arxiv.org/abs/2610.07591) feeds a transformer's output back into itself, which allows the model to perform more compute on the same number of parameters. There are rumors (unconfirmed) that OpenAI has used this approach for their GPT-6 line of models.

I won't dive too deep into the details of the RNN architecture that Graves used, because that's already done better elsewhere: Distill's ["Four Experiments in Handwriting with a Neural Network"](https://distill.pub/2016/handwriting/) is a great interactive walkthrough.

I did, however, need to make significant changes to Graves' architecture in order to accommodate Hangul. If you're interested in the details, I've described some of these changes in [a separate post](/graves-departures).

### Handwriting models and Large Language Models

A Large Language Model (LLM) and a handwriting model are very different architectures. However, they have a lot in common. First of all, they are both producing a set of sequences. And in most cases, they are doing this auto-regressively.

Secondly, they both use a form of attention. You have probably heard about the transformer, which was introduced in the 2017 paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). The transformer architecture directs attention across an entire context window of tokens. This allows the forward pass to model a complex relationship between the next token and all the tokens that came before it.

But attention existed as a concept before this 2017 paper. In 2013, Alex Graves introduced a type of attention that is optimized for a handwriting model. Instead of attending to the entire context window (like a transformer), Graves' attention mechanism sweeps monotonically across the text embedding as the model draws text. When generating strokes, we really only care about the current character, and the nearby characters, so this system works well, and is much more efficient than a full transformer.

## Training Data

Graves' model was trained on about 5,400 lines of English handwriting from 221 different writers. I was not able to find any comparable data source for Hangul, so I had to create my own.

I trained this model on about 4,500 lines of handwritten text that I collected from a [Neo Smartpen](https://shop.neosmartpen.com/products/neo-smartpen-a1), all done by a single writer, Jo YW, whom I found on soomgo.com. Every stroke you see in the demo is the model's imitation of his hand.

I chose the Neo Smartpen for several reasons.

The Neo Smartpen is created by a Korean company, so it fits spiritually well with this project.

More importantly, though, the Neo Smartpen is very hackable. The company provides an [open source example app](https://github.com/NeoSmartpen/WEB-SDK-Sample) on GitHub, built on their [Web SDK](https://github.com/NeoSmartpen/WEB-SDK2.0), that shows how to interact with the pen over the Web Bluetooth API.

As the writer uses the Smartpen, the device feeds data to the Chrome window. It includes the x, y position, the pen pressure, the pen up/down state, and some other attributes that I'll ignore for this post.

The pressure turned out to matter more than I expected. Graves' model only saw pen position, and my first version did too. When I added pressure as an input and an output, the model learned noticeably faster on less data. In hindsight that makes sense. Pressure is a strong hint about where you are within a stroke. It climbs at the start, holds through the middle, and tails off before the pen lifts.

At first, I started by having Jo write handwriting strokes from text data that I sourced from Wikipedia. I wanted to start with natural data that was unambiguously unencumbered by copyright. This worked well enough for me to train a proof-of-concept of the model, however, I noticed the model had many holes. For example, any sentence that ended with a 요 (a spoken, polite ending) would tend to hallucinate. This makes sense, because Wikipedia is written using the neutral dictionary form, and almost never ends a sentence with 요.

From that point on, I decided to switch from Wikipedia text to generating synthetic data from Claude. After each training run, I would identify specific holes in the model, and then create more synthetic data for Jo to write. This worked well. Jo also found it more enjoyable. If I was starting again, I might go the straight synthetic route, rather than bootstrapping with Wikipedia text.

I've published all the training data on Hugging Face: [jbaudanza/korean-handwriting-strokes](https://huggingface.co/datasets/jbaudanza/korean-handwriting-strokes). It's released under CC BY-SA 4.0, so you're free to use it for your own models, art, or teaching.

## Future Plans

I'm probably not going to do much more with this, since I'd like to focus more now on audio models. However, I have some ideas if anyone wants to try something. You're free to use all the [source code](https://github.com/Hilokal/korean-handwriting) (MIT) and the data and model weights (CC BY-SA 4.0) that I've collected:

### Multiple writers

This model is trained on data from only one writer. If you had data from multiple writers, it opens up more possibilities for model capabilities.

For example, you could train some kind of style-embedding feature. You could perhaps "clone" someone's handwriting, the same way you can clone someone's voice with a text-to-speech model.

### Flow Matching

My model uses something called a mixture density network (MDN) to predict the next pen stroke. Basically, this means it's predicting several probability density functions, which you can sample from.

However, many state-of-the-art models use a newer technique called "flow matching". Instead of sampling from a probability distribution, you start with a random point, and then the model predicts a directional vector from this random point. You do this iteratively until you arrive at the final result. This is how most image generation models work.

### Different scripts and alphabets

There are hundreds of active alphabets in the world. I'm particularly interested in Korean, but I encourage anyone to explore modeling different alphabets as well.

### Drive a real mechanical pen

I think it would be fun to hook the output of this model up to a real pen. I'm not sure of the use case, but maybe you're an artist and have some ideas?

### Art collaboration?

Are you a Korean artist that is interested in doing some sort of collaboration? If so, please [send me an email](mailto:jon@jonb.org).

In the meantime, anything you generate in the demo above can be downloaded as an animated SVG (or as raw stroke JSON) using the links under the canvas.
