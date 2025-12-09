# Presentation Script: Autonomous Rocket Landing
## 3-5 Minute Non-Technical Demo Video

**Target Audience:** Non-technical viewers (general public, potential investors, stakeholders)  
**Tone:** Engaging, accessible, inspiring  
**Timing:** ~4 minutes total

---

## [SLIDE 1: Title Slide] (5 seconds)

**[Pause for title to display]**

"Hi, I'm [Name], and this is [Name]. Today we're excited to share our work on autonomous rocket landing using artificial intelligence."

---

## [SLIDE 2: The Real-World Problem] (35 seconds)

"Landing a rocket safely is one of the most challenging problems in aerospace engineering. Think about it: a rocket is falling from the sky, moving incredibly fast, and it needs to land precisely on a small landing pad—all while using as little fuel as possible. Modern Aerospace companies like SpaceX are challenged with this problem regularly.

This isn't just a video game challenge. This is the real problem that companies like SpaceX and Blue Origin face every time they want to reuse a rocket. The three key objectives are simple: safety—don't crash, fuel efficiency—don't waste resources, and precision—hit the target.

Right now, this requires teams of engineers writing thousands of lines of code. But what if we could teach an AI to learn how to do this on its own?"

---

## [SLIDE 3: Our Unified Goal] (25 seconds)

"Our project goal was to develop and compare different AI algorithms that can autonomously land a rocket safely and efficiently.

This addresses a critical challenge in aerospace engineering that's directly relevant to the future of space exploration. Whether it's landing on Mars, returning from the Moon, or making space travel more affordable through reusable rockets—autonomous landing systems are essential."

---

## [SLIDE 4: What We Built] (40 seconds)

"We built an AI-powered rocket landing system—an intelligent agent that learns to land rockets autonomously through practice, just like a human pilot would.

While our earlier coursework involved basic AI on simple problems, this project represents a significant step forward. We implemented advanced deep learning techniques including experience replay—where the AI learns from past experiences, not just current ones—target networks for stable learning, and comprehensive evaluation across thousands of training episodes.

Here's what our system can do: It starts from random positions in the sky, navigates to the landing pad, controls the rocket engines to slow down and steer, and most importantly—it learns fuel-efficient strategies over time through rigorous training and testing.

You can see it in action right now. [If showing demo: Point to screen] Watch as the rocket starts high in the air, adjusts its trajectory, and makes a smooth landing. This isn't pre-programmed—the AI learned this behavior through extensive practice, with proper validation and testing to ensure reliability."

---

## [SLIDE 5: Two AI Approaches] (40 seconds)

"We didn't just build one solution—we compared two different AI methods to see which works better. This goes well beyond basic implementations.

The first approach, called DQN, uses advanced techniques like experience replay—where the AI learns from thousands of past experiences stored in memory—and target networks that provide stable learning signals. Think of it like this: imagine you're trying to learn which actions are most valuable. Every time you try something, you remember whether it worked well or poorly, and over time you build up a mental map of what to do in different situations. But here, the AI can also learn from experiences it had hours or days ago, not just what happened right now.

The second approach, called A2C, uses separate networks—one for making decisions and one for evaluating those decisions. It's more direct—it learns a policy, like a set of rules: 'If I'm high and moving fast, do this. If I'm low and near the pad, do that.' But it does this with sophisticated neural networks that work together.

The exciting part? Both methods successfully learned to land rockets through extensive training. But as we'll see, one performed significantly better."

---

## [SLIDE 6: Performance Results] (50 seconds)

"Let's look at the results. We rigorously tested our AI systems on separate validation and test sets—proper scientific evaluation, not just cherry-picked results. We tested each system on 50 different landing attempts across different scenarios, and here's what we found:

Our best system—DQN with Adam—achieved a perfect 100% success rate. Every single landing attempt was successful. It also used fuel very efficiently, averaging about 76 units per landing.

The other DQN variant, using RMSprop, also achieved 100% success, though it used slightly more fuel.

The A2C method with Adam achieved a 64% success rate—still impressive, but not perfect. However, it was the most fuel-efficient, using only 58 units of fuel on average.

These results come from comprehensive evaluation with proper train-test splits, tracking multiple metrics including success rates, crash rates, fuel consumption, and episode returns. The key takeaway? We've proven that AI can learn to land rockets reliably, with some methods achieving perfect success rates through rigorous training and evaluation."

---

## [SLIDE 7: Optimizer Comparison] (35 seconds)

"Let me explain why we tested different optimizers—think of them as different learning strategies.

For DQN, we compared Adam and RMSprop—both achieved perfect 100% success rates, but Adam was more fuel-efficient, using 20% less fuel.

For A2C, the difference was dramatic. Adam achieved a 64% success rate, while SGD—another learning strategy—only achieved 28% success and used four times more fuel. This shows that choosing the right learning method is crucial—some approaches simply don't work well for certain problems.

This kind of comparison helps engineers understand which AI techniques to use in real-world applications."

---

## [SLIDE 8: Why This Matters] (35 seconds)

"So why does this matter? The applications are everywhere.

First, space exploration. Autonomous spacecraft landing is essential for future missions to the Moon and Mars, where human pilots can't be present.

Second, autonomous systems. The same techniques power drone navigation, precision control systems, and other technologies that need to operate safely without human intervention.

Third, cost reduction. Reusable rocket technology is only economically viable if landing is reliable and automated. Every successful landing saves millions of dollars."

---

## [SLIDE 9: Broader Impact] (25 seconds)

"The same AI techniques we used here power many technologies you interact with every day: advanced robotics systems, autonomous vehicles, industrial automation, and even game AI.

By showing that AI can learn to balance safety and efficiency in such a complex task, we're advancing the state of autonomous systems. This research contributes to a future where AI can handle critical tasks reliably and safely."

---

## [SLIDE 10: Summary] (35 seconds)

"Let's summarize what we've accomplished.

We've shown that reinforcement learning—a type of AI that learns through trial and error—can successfully solve the rocket landing problem. Our best system achieved a perfect 100% success rate while maintaining fuel efficiency.

We compared two different AI approaches and multiple learning strategies, providing valuable insights for real-world applications.

This work has real-world relevance for aerospace companies, contributes to AI research, and demonstrates the potential of autonomous systems. The code is open-source, making it available for educational use and further research."

---

## [SLIDE 11: Thank You] (10 seconds)

"Thank you for watching. We're Ryan Christ and Jay Parmar from Duke University, and this is our CS 372 Final Project.

You can find our project repository and all the code on GitHub. We hope this demonstrates the exciting potential of AI in solving complex real-world problems.

Thank you!"

---

## **TOTAL TIME: ~4 minutes 30 seconds**

---

## **Presentation Tips:**

1. **Visual Emphasis:** When showing the performance results slide, point to the numbers and emphasize the 100% success rate.

2. **Demo Timing:** If including a live demo, pause the script and let the landing animation play (10-15 seconds), then resume with "As you can see, the AI successfully navigates and lands."

3. **Tone Variations:**
   - Start with enthusiasm and curiosity
   - Build excitement during results
   - End with confidence and impact

4. **Pacing:** 
   - Don't rush through the results slide—this is the key moment
   - Pause briefly after major points to let them sink in
   - Speak clearly and at a moderate pace

5. **Visual Cues:**
   - Point to specific numbers on the results table
   - Use hand gestures to emphasize "perfect 100%" or "four times more fuel"
   - Make eye contact with the camera (not the slides)

6. **Adaptation:** If you need to shorten to 3 minutes, reduce the optimizer comparison and broader impact sections. If extending to 5 minutes, add more detail about the learning process or real-world examples.

