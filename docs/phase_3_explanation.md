# Phase 3: Cognitive Scheduler & I/O Subsystem (in layman terms)

## The Big Picture

Phase 1 taught the computer to "read comprehension" - understanding how words group into ideas.

Phase 2 gave the computer "smart memory management" - deciding what to keep in fast vs. slow memory.

**Phase 3 enables multi-tasking with tools** - the computer can now:
- Work on multiple thoughts at once (multi-threaded reasoning)
- Use external tools (calculator, search, etc.) without losing its place
- Switch between tasks while preserving mental state

Think of it like how you work:
- You might be writing an essay (main task)
- Then pause to look up a fact (tool use)
- Then return to writing, remembering exactly where you were

---

## The 5 Main Components

### 1. Reasoning Control Block (RCB) = "Thread's Notebook"

Each "thought process" (thread) has its own notebook tracking:

```
┌─────────────────────────────────────┐
│     REASONING CONTROL BLOCK         │
├─────────────────────────────────────┤
│ Thread ID: thread_1_abc12345       │
│ Priority: HIGH (important!)         │
│ State: READY to run                │
│                                     │
│ Currently thinking about:           │
│   - Active slice: "paragraph 3"     │
│   - Context: ["intro", "para 2"...] │
│                                     │
│ Tool calls waiting:                 │
│   - Calculator: "2 + 2" (done)     │
│   - Search: "neural networks"       │
│                                     │
│ How long has it run? 523ms         │
│ How "focused" is it? 0.85           │
└─────────────────────────────────────┘
```

The RCB is like a worker's desk tag + notebook:
- **Who they are**: Thread ID
- **How important**: Priority (CRITICAL > HIGH > NORMAL > LOW)
- **What they're doing**: Attention focus
- **What tools they're using**: Tool call stack
- **How long they've been working**: Runtime tracking

---

### 2. Cognitive Scheduler = "Project Manager"

The Cognitive Scheduler decides which thread gets to use the CPU:

```
┌─────────────────────────────────────┐
│        COGNITIVE SCHEDULER           │
│                                     │
│  3 threads waiting:                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐│
│  │ Thread 1│ │ Thread 2│ │ Thread 3││
│  │ HIGH    │ │ NORMAL  │ │CRITICAL ││
│  └─────────┘ └─────────┘ └─────────┘│
│                                     │
│  Who goes first?                    │
│  → Thread 3 (CRITICAL priority) ✓   │
│                                     │
│  Scoring formula:                   │
│  score = priority + focus + wait     │
└─────────────────────────────────────┘
```

**How it decides:**
1. **Priority first**: Safety-critical tasks always win
2. **Cognitive fidelity**: Threads that are "focused" get preference
3. **Fairness**: Threads waiting long get a boost (avoid starvation)

**Real-world analogy:**
- Like a restaurant kitchen with multiple orders
- Critical orders (allergies!) go first
- Then by who has been waiting longest
- Chef can only work on one dish at a time

---

### 3. I/O Peripheral Registry = "Toolbox"

External tools are registered like devices:

```
┌─────────────────────────────────────┐
│      I/O PERIPHERAL REGISTRY         │
├─────────────────────────────────────┤
│                                     │
│  🌐 web_search                      │
│    → Search the internet            │
│    → Type: SEARCH                   │
│                                     │
│  🔢 calculator                     │
│    → Do math calculations           │
│    → Type: CALCULATOR               │
│                                     │
│  📊 text_analyzer                   │
│    → Analyze sentiment              │
│    → Type: CLASSIFIER               │
│                                     │
│  📁 file_writer                     │
│    → Save to disk                  │
│    → Type: FILE_WRITER              │
└─────────────────────────────────────┘
```

**How tools are called:**
1. Thread says "I need to calculate something"
2. Triggers TOOL_CALL interrupt
3. Scheduler saves thread's state
4. Tool executes (calculator runs)
5. Result returned to thread
6. Thread resumes where it left off

**Key insight**: The thread doesn't just "call" the tool - it creates an **interrupt** that pauses the thread, runs the tool, and then resumes.

---

### 4. Interrupt Vector Table (IVT) = "Switchboard"

Maps different types of events to their handlers:

```
┌─────────────────────────────────────────────────────────────┐
│                  INTERRUPT VECTOR TABLE                     │
├────────┬──────────────┬───────────┬─────────────────────────┤
│ Vector │ Type         │ Priority  │ Description             │
├────────┼──────────────┼───────────┼─────────────────────────┤
│  0x01  │ TOOL_CALL    │    10     │ Tool wants to run        │
│  0x02  │ TOOL_RESULT  │     5     │ Tool finished           │
│  0x03  │ TOOL_ERROR   │     3     │ Tool crashed            │
│  0x11  │ PAGE_FAULT   │     8     │ Need memory from disk    │
│  0x20  │ TIME_SLICE   │     7     │ Thread's time is up      │
│  0x22  │ PREEMPT      │     2     │ Important thread ready   │
│  0x31  │ SHUTDOWN     │     1     │ System shutting down     │
│  0x32  │ ERROR        │     0     │ CRITICAL ERROR!          │
└────────┴──────────────┴───────────┴─────────────────────────┘

Lower priority number = more important!
0 = highest priority (ERROR - handle immediately!)
```

**How it works:**
1. Something happens (tool needs to run)
2. Interrupt generated (e.g., TOOL_CALL at 0x01)
3. IVT routes to correct handler
4. Handler does its job
5. System resumes

**Analogy**: Hospital emergency room
- Priority 0: Heart attack (handle immediately!)
- Priority 1: Broken bone
- Priority 10: Routine checkup

---

### 5. Reasoning Interrupt Cycle (RIC) = "Task Switcher"

The RIC manages switching between threads when interrupts happen:

```
┌─────────────────────────────────────────────────────────────┐
│             REASONING INTERRUPT CYCLE                       │
│                                                             │
│  1. Thread needs tool                                      │
│      ↓                                                      │
│  2. INTERRUPT: "Hey, I need calculator!"                  │
│      ↓                                                      │
│  3. RIC receives interrupt                                  │
│      ↓                                                      │
│  4. Save thread's mental state                              │
│     - What was I thinking?                                  │
│     - Where was I in the sentence?                          │
│     - What tools was I using?                               │
│      ↓                                                      │
│  5. Block thread (go to waiting room)                       │
│      ↓                                                      │
│  6. Run the tool                                            │
│     - Calculator: "2 + 2 * 10 = 22"                        │
│      ↓                                                      │
│  7. Perception Alignment (format result)                     │
│     - Convert to: {tool: "calculator", value: 22}          │
│      ↓                                                      │
│  8. Unblock thread with result                              │
│      ↓                                                      │
│  9. Thread resumes exactly where it left off!              │
└─────────────────────────────────────────────────────────────┘
```

**Perception Alignment** - A key innovation!

When a tool returns raw data, Perception Alignment transforms it into the cognitive model's "language":

```
Raw calculator output:
  22

After Perception Alignment:
  {
    "type": "tool_result",
    "tool_name": "calculator",
    "content": 22,
    "timestamp": "2026-02-28T13:12:59.631762"
  }
```

This is like a translator:
- Tool speaks "machine language"
- Cognitive model speaks "semantic language"
- Perception Alignment bridges the gap

---

## Why Does This Matter?

Traditional LLMs have a big limitation: they can't really multitask.

```
Without AgentOS:
  User: "Calculate 2+2*10"
  LLM: "22"
  User: "Now search for AI news"
  LLM: [forgets previous context]

With AgentOS (Phase 3):
  Thread 1: "Calculating..." [uses calculator tool]
  Thread 2: "Searching..." [uses search tool]
  Both threads maintain their state!
```

**Real-world impact:**

1. **Multi-step reasoning**: Can chain multiple tools together
2. **Long-running tasks**: Background work doesn't block everything
3. **Priority handling**: Critical tasks jump the line
4. **State preservation**: Never lose your place in a thought

---

## Analogy: A Busy Kitchen

```
┌─────────────────────────────────────────────────────────────┐
│                    THE KITCHEN                              │
│                                                             │
│  Chef (CPU) can only do one thing at a time                 │
│                                                             │
│  Orders in queue:                                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Order 1: Table 5's main dish (CRITICAL - hungry!)    │   │
│  │ Order 2: Table 3's dessert (HIGH)                    │   │
│  │ Order 3: Table 8's appetizer (NORMAL)                │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  Chef needs a tool (mixer) → INTERRUPT!                     │
│  → Sous-chef brings mixer                                   │
│  → Chef uses it, returns to cooking                         │
│  → Order 1 continues cooking                                │
│                                                             │
│  This is EXACTLY how the Cognitive Scheduler works!         │
└─────────────────────────────────────────────────────────────┘
```

---

## What Got Built

```
┌──────────────────────────────────────────────────────────────┐
│                 COGNITIVE SCHEDULER & I/O SUBSYSTEM          │
│                                                              │
│  ┌──────────────┐   ┌──────────────────┐  ┌──────────────┐   │
│  │   Threads    │   │  Interrupt       │  │   Tools      │   │
│  │              │   │   Vectors        │  │              │   │
│  │ Thread 1     │   │  0x01 TOOL_CALL  │  │ Calculator   │   │
│  │ Thread 2     │   │  0x02 TOOL_RESULT│  │ Web Search   │   │
│  │ Thread 3     │   │  0x20 TIME_SLICE │  │ Text Analyzer│   │
│  │              │   │  0x22 PREEMPT    │  │              │   │
│  └──────┬───────┘   └────────┬─────────┘  └──────┬───────┘   │
│         │                    │                   │           │
│         │      ┌─────────────┴──────────────────────┐        │         
│         │      │         COGNITIVE SCHEDULER        │        │         
│         │      │   (Decides who runs next)          │        │         
│         │      └─────────────┬──────────────────────┘        │         
│         │                    │                               │         
│         │      ┌─────────────┴──────────────┐                │         
│         │      │    REASONING INTERRUPT     │                │         
│         │      │    CYCLE (RIC)             │                │         
│         │      │                            │                │         
│         │      │  Save → Block → Execute    │                │         
│         │      │  → Align → Resume          │                │         
│         │      └────────────────────────────┘                │         
│         │                                                    │         
│         └────────────────────────────────────────────────────│         
│                                                              │         
│  Features:                                                   │         
│  • Multi-threaded reasoning (3+ threads at once)             │         
│  • Priority-based scheduling (CRITICAL wins)                 │         
│  • Tool use without losing state                             │         
│  • Fast context switching (0.003 ms)                         │         
│  • Interrupt-driven architecture                             │         
└──────────────────────────────────────────────────────────────┘
```

---

## Connection to Previous Phases

```
Phase 1: Break text into semantic slices (σ₁, σ₂, σ₃)
           ↓
Phase 2: Decide which slices go in fast vs. slow memory
           ↓
Phase 3: Manage multiple threads using those slices,
          with tool use and interrupt-driven switching
           ↓
Phase 4: Multiple agents sharing memory and syncing up
```

Each phase builds on the previous one - like layers of an operating system!
