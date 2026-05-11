# Inside the AI's Brain 🧠

Ever wonder what an AI model actually *looks* like under the hood? It's not magic, it's just math! This page breaks down the "Transformer" architecture that powers Fragment LLM in plain English. 

We built this model to be modern, incredibly fast, and capable of scaling from a tiny brain you can run on a laptop to a massive 7B-parameter giant.

---

## How Does It Think? (The Big Picture)

Imagine the AI is trying to read a sentence and guess the next word. It goes through a series of steps:

```
1. Reads Your Text
        ↓
2. Looks Up the Words (Token Embedding)
   Translates your English words into numbers it understands.
        ↓
3. The "Thought Process" (Transformer Blocks)
   It passes the numbers through several layers of math. 
   At each layer, it asks: "What do these words mean in context?"
   "Which words are related to each other?"
        ↓
4. The Final Guess (Language Model Head)
   After thinking about it, it picks the single most likely 
   word to come next!
```

---

## The Core Ingredients

If you look in our `src/model.py` file, you'll see the exact Python code that makes up the brain. Here is what each piece actually does:

### 1. Token Embedding (The Dictionary)
The AI doesn't know what an "apple" is. It only knows numbers. The Token Embedding is basically a giant dictionary. When you feed it a word, it looks up a unique string of numbers (a vector) that represents that word's "meaning."

### 2. Rotary Position Embeddings (RoPE) 🧭
If you give the AI the words "The dog bit the man" and "The man bit the dog", the words are exactly the same! The AI needs to know what *order* they are in. 
Instead of just tagging words with a simple position number, we use something called **RoPE** (Rotary Position Embeddings). It's a clever math trick that rotates the word's meaning based on its position. This helps the AI understand sentences of any length much better than older models.

### 3. Grouped-Query Attention (GQA) 👁️
When you read a long paragraph, you pay "attention" to certain important words to understand the context. The AI does this too!
Older AI models stored a massive, separate list of memories for every single word they read, which ate up tons of RAM. We use **Grouped-Query Attention (GQA)**. Think of it like a group of students sharing one textbook instead of everyone buying their own. It gets the same job done but uses way less memory, which means the AI generates text *much* faster.

### 4. SwiGLU Feed-Forward Network ⚡
After the AI pays "attention" to the words, it has to process what it learned. It passes the information through a Feed-Forward Network. 
Most models use a standard math equation here, but we use **SwiGLU**. It sounds like a strange brand of glue, but it's actually just a highly optimized equation that empirically makes the AI learn faster and perform better.

### 5. RMSNorm (The Stabilizer) ⚖️
As numbers get passed through dozens of layers of math, they can get wildly huge or microscopically small. We use **RMSNorm** to quickly scale the numbers back to a normal size after every step. It's like a fast-acting volume knob that keeps the math from blowing out the speakers.

### 6. The Language Model Head (The Guesser) 🎯
Once the numbers have made it all the way through the brain, they reach the end. The Language Model Head looks at the final processed numbers and calculates the probability for *every single word in its dictionary*. It then spits out the highest-probability word as its answer!

*(Fun fact: We use a trick called "Weight Tying". The giant dictionary we used in Step 1 to turn words into numbers? We just run it in reverse here to turn the final numbers back into words. Re-using it saves us millions of parameters!)*

---

## The Secret Sauce: Why Is It So Fast?

If you try to run standard AI models on a regular computer, they usually crash. Here's a summary of the modern shortcuts we took to make Fragment LLM run smoothly:

1. **RMSNorm**: A stripped-down, faster version of the stabilizer older AIs use.
2. **RoPE**: Better handling of long sentences without needing extra memory.
3. **GQA (Grouped Query Attention)**: Shares memory across the attention heads so generating text doesn't fill up your RAM.
4. **Flash Attention**: A highly optimized chunk of code (written by NVIDIA/PyTorch) that calculates attention incredibly fast on your GPU.
5. **SwiGLU**: A slightly more complex equation that results in a noticeably smarter AI.
6. **Gradient Checkpointing**: A trick used during training. Instead of memorizing all the intermediate math steps (which takes tons of RAM), it throws them away and just quickly recalculates them when needed.

---

**Where to next?**  
Ready to actually train this brain? Head over to the [Training Guide](training.md)!
