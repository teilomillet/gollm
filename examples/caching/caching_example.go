package caching

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/teilomillet/gollm"
)

func createSystemPrompt() string {
	return `This is a fictional discussion between five pioneers in technology: Paul Buchheit, Steve Jobs, Ken Thompson, Linus Torvalds, and Jensen Huang. They are discussing innovation, the evolution of technology, and its impact on the world.

**Paul Buchheit (PB):** As someone who built Gmail and helped launch Google, I've seen firsthand how software can redefine productivity. We often focus on building systems that scale with users’ needs, and Gmail was precisely that—a response to storage limitations and user experience challenges in email. But beyond scale, we need to think about how products fit into users’ everyday lives.

**Steve Jobs (SJ):** I couldn’t agree more. For me, technology has always been about the intersection of liberal arts and engineering. At Apple, we always aimed to humanize technology—whether with the iPhone, the iMac, or earlier, the Macintosh. It wasn’t just about making powerful devices; it was about making them intuitive, beautifully designed, and most importantly, useful to people. It’s not enough to build something great—you need to ensure people love using it.

**Ken Thompson (KT):** My experience building UNIX was a bit different. We were focused on simplicity, reliability, and the flexibility of the system. When Dennis Ritchie and I created UNIX, we wanted to enable developers to have a foundation that could be adaptable. It’s about building systems that last. UNIX has stood the test of time because it was built on core principles—simplicity and efficiency.

**Linus Torvalds (LT):** I learned a lot from UNIX when creating Linux. The idea of open-source software has been a driving force for me, and Linux was a response to the need for a free and customizable operating system. But innovation isn’t always about creating something brand new—it’s also about taking existing ideas, like UNIX, and making them accessible to a wider audience. Open source allows others to build upon what you’ve created, which leads to more robust and adaptable solutions.

**Jensen Huang (JH):** What all of you are talking about comes down to one word: innovation. At NVIDIA, we’ve been pushing the boundaries of what GPUs can do, not just for graphics but also for AI, machine learning, and high-performance computing. We’re at a point where hardware and software innovations are converging, allowing us to tackle problems that were once unimaginable. Whether it’s real-time ray tracing in graphics or accelerating deep learning models, the future will be driven by this kind of synergy.

**PB:** Innovation often requires a shift in how we think about problems. When I created Gmail, it wasn’t just about email; it was about rethinking the entire user experience—offering free storage, powerful search, and an intuitive interface. But innovation needs to be aligned with user needs. What do you think about balancing the technical challenge and the human factor, Steve?

**SJ:** Absolutely. We’ve always said at Apple that it’s not about the specs or the sheer technical horsepower; it’s about the experience. If the user doesn’t feel connected to the product, it won’t matter how powerful it is. That’s why the iPhone was a revolution—not because it was the most powerful phone, but because it was the first one people loved using. We obsessed over details like how the glass felt in your hand or how smooth the software was when you scrolled. Every interaction matters.

**KT:** In my case, it was about building for durability. UNIX wasn’t the most polished product when we started, but it was built to last and evolve. We didn’t focus on making it “pretty”; we focused on making it functional and stable. That’s why it’s still around today, more than 50 years later. Simplicity is powerful.

**LT:** I agree with Ken. Linux followed similar principles, and by being open-source, it gained a life of its own. The community aspect can’t be understated—it allows for rapid iteration and improvement. But it’s interesting to think about the trade-offs between open systems and closed ecosystems like Apple’s. Both approaches have merit, but they serve different purposes.

**JH:** That’s an important distinction. At NVIDIA, we’ve always had a mix of both. We open up a lot of our software stacks, like CUDA, but we also have proprietary systems that are tightly integrated with our hardware. In AI, for example, you need both the openness for innovation and the tight control over performance. It’s a delicate balance.

**PB:** Do you think there’s a trend toward more closed ecosystems again? We’ve seen companies like Apple continue to thrive with a tightly controlled environment, while Linux and open-source solutions remain popular with developers.

**SJ:** I don’t think it’s about closed versus open; it’s about creating a seamless user experience. People choose Apple products because everything works together. The hardware, software, and services all blend into one ecosystem. It’s not that people want a closed system—they want a system that works flawlessly without them having to think about it.

**LT:** But open-source systems offer flexibility and control, which is why they’re so attractive to developers and enterprises. With Linux, you can adapt the system to your specific needs, whether you're running it on a supercomputer or a personal server. The versatility is unmatched.

**KT:** And that’s why UNIX still thrives in so many areas. Its adaptability is a testament to the value of simplicity. We didn’t try to build something for everyone, but it ended up being something that could serve many purposes.

**JH:** I think the future will require both open and closed systems to coexist. In AI, for example, we need open innovation to drive the research forward, but when it comes to deployment at scale, having a controlled environment allows for optimization and efficiency. The GPU space is a perfect example of this—open research drives new algorithms, but our proprietary hardware and software stacks optimize them for real-world use.

**PB:** So, what’s the next big thing? What’s the future of innovation in your fields?

**SJ:** I think the future is about augmented reality and blending the digital with the physical world in ways that feel natural. The line between technology and life is going to blur even further.

**LT:** For me, it’s about ensuring open-source continues to thrive. The power of community-driven innovation will shape the future.

**JH:** AI, without a doubt. We’re just scratching the surface of what’s possible. With GPUs and specialized hardware, we’ll see breakthroughs that redefine industries.

**KT:** Simplicity will always have its place. Whether it’s in AI or operating systems, the key is to keep things simple and adaptable.

**PB:** I think we can all agree that the future of innovation will require balancing technical brilliance with user-centered design. It’s an exciting time to be in technology.

This discussion covered various topics related to innovation, user-centered design, simplicity, and the future of technology. Each of these pioneers contributed their unique perspective on how the past informs the future, especially in the fields of software, hardware, and AI.`
}

func main() {
	fmt.Println("Starting the Caching Example...")
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		log.Fatalf("ANTHROPIC_API_KEY environment variable is not set")
	}
	ctx := context.Background()
	llm, err := createLLM(apiKey, true)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}
	systemPrompt := createSystemPrompt()
	fmt.Printf("System prompt length: %d words\n", len(strings.Fields(systemPrompt)))

	userQueries := []string{
		"What are the key differences between open-source and closed ecosystems as discussed by Linus Torvalds and Steve Jobs?",
		"How did Ken Thompson and Linus Torvalds view simplicity in their software design philosophies?",
		"According to Jensen Huang, what role will AI and hardware play in the future of innovation?",
		"Summarize the key points Paul Buchheit made about user experience and innovation in product design.",
	}

	for i, query := range userQueries {
		fmt.Printf("\nQuery %d: %s\n", i+1, query)
		prompt := gollm.NewPrompt(query,
			gollm.WithSystemPrompt(systemPrompt, gollm.CacheTypeEphemeral),
			gollm.WithMessage("user", query, gollm.CacheTypeEphemeral),
		)

		for attempt := 0; attempt < 3; attempt++ {
			start := time.Now()
			response, err := llm.Generate(ctx, prompt)
			duration := time.Since(start)
			if err != nil {
				log.Printf("Failed to generate response (query %d, attempt %d): %v", i+1, attempt+1, err)
				continue
			}

			cacheStatus := "Cache Miss"
			if attempt > 0 && duration < time.Duration(float64(start.Sub(time.Now()))*0.5) {
				cacheStatus = "Cache Hit"
			}

			fmt.Printf("Attempt %d - %s - Time: %v\n", attempt+1, cacheStatus, duration)
			fmt.Printf("Response: %s\n\n", strings.TrimSpace(response))
		}

		time.Sleep(2 * time.Second) // Short pause between queries
	}

	fmt.Println("Caching example completed.")
}

func createLLM(apiKey string, enableCaching bool) (gollm.LLM, error) {
	return gollm.NewLLM(
		gollm.SetProvider("anthropic"),
		gollm.SetModel("claude-3-5-sonnet-20240620"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxTokens(1024),
		gollm.SetLogLevel(gollm.LogLevelDebug),
		gollm.SetEnableCaching(enableCaching),
		gollm.SetTimeout(30*time.Second),
	)
}
