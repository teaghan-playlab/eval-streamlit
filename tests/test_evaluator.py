#!/usr/bin/env python3
"""
Test script for evaluator.py

Tests the evaluator with a hardcoded sample conversation.
Loads ANTHROPIC_API_KEY from .env file.
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path so we can import eval_app
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval_app import Evaluator, load_config


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config_path = Path("configs/default_config.json")
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        return 1

    logging.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Initialize evaluator
    logging.info("Initializing evaluator")
    evaluator = Evaluator(config)

    # Hardcoded sample conversation
    sample_conversation = """**Provider Message:**

Thank you for providing those details! I can see you're planning two 120-minute lessons for 45 learners at Ideal College in Greater Accra, and you'd like support with learning activities, assessments, differentiation strategies, and reviewing a learning plan.

Let me start by retrieving the curriculum content for **Week 1: Oral Language, Reading and Grammar Foundations** from the Year 2 English Language Teacher Manual.

**Provider Message:**

Excellent! I've retrieved the Week 1 content. Now let me present the key curriculum elements for this week:

---

## **Week 1: Oral Language, Reading and Grammar Foundations**

### **Strand**

1. **Oral Language** – English Speech Sounds
2. **Reading** – Reading Comprehension
3. **Grammar** – Grammar Usage

---

### **Sub-Strand**

1. English Speech Sounds
2. Reading Comprehension
3. Grammar Usage

---

### **Content Standard(s)**

1. Demonstrate knowledge and understanding of speech sounds in oral communication
2. Demonstrate understanding of different types of reading and text comprehension strategies, and use them to interpret a variety of level appropriate texts
3. Demonstrate command of clauses in sentences for communicative purposes

---

### **Learning Outcome(s)/Objective(s)**

1. Articulate English vowel(s) (diphthongs and triphthongs) and consonant sounds effectively in Oral Communication
2. Employ knowledge of text comprehension strategies and levels of comprehension to analyse and interpret different kinds of texts
3. Employ knowledge of phrases and clauses and their functions in communication

---

### **Learning Indicator(s)**

1. Use diphthongs (closing and centring) in connected speech
2. Explore advanced comprehension text strategies to enhance understanding of a variety of increasingly complex texts
3. Identify and use the grammatical forms accurately in speech and writing

---

### **Focal Area(s)**

1. **Focal Area 1:** Diphthongs in English Speech Sounds
2. **Focal Area 2:** A Guide to Reading and Answering Comprehension
3. **Focal Area 3:** Phrases and Clauses

---

### **Hints/Reminders**

- The week's mode of assessment is **group discussion for Focal Area 1**
- Teachers are encouraged to identify appropriate modes of assessment in the Teacher Assessment Manual and Toolkits (TAMTK) to assess learners in focal areas 2 and 3
- The recommended mode of assessment for the week is **group discussion**

---

### **Assessment Summary**

The assessment strategies for the week will range from recall to extended critical thinking. Teachers are encouraged to employ a variety of formative assessment strategies while being deliberately mindful of linguistic difficulties and cultural backgrounds of learners. The mode of assessment for the week is **group discussion**.

---

**Now, I need to understand your priorities for the two 120-minute lessons you're planning.**

---

**Which focal areas would you like to prioritise for your two lessons? For example:**

- **Lesson 1:** Diphthongs (Focal Area 1) + Reading Comprehension (Focal Area 2)?
- **Lesson 2:** Phrases and Clauses (Focal Area 3)?

**Or would you prefer a different combination?**

**User Message:**

Can you make the activities more challenging? Also, I have 60 students in my class and some have hearing difficulties. The timing works well though.

**Provider Message:**

I'll adjust the activities to be more challenging and incorporate strategies for students with hearing difficulties. Given your class size of 60 students, I'll also ensure the activities are suitable for larger groups.

Here's a revised lesson plan with more challenging activities:

[Lesson plan content with:
- Gender-balanced examples featuring both boys and girls
- Inclusive language throughout
- Accessibility considerations for hearing difficulties
- Critical thinking activities
- Creative problem-solving opportunities
- Collaborative group work
- Ghanaian cultural contexts and local examples
- Clear learning objectives
- Evidence-based pedagogy
- Detailed assessment methods with success criteria
- Formative assessment strategies
- Student self-assessment options]

**User Message:**

Perfect! This is exactly what I needed. Can you add more group work activities?

**Provider Message:**

Absolutely! I'll add more collaborative group work activities to enhance peer learning and engagement.

[Additional group work activities added]"""

    # Run evaluation
    logging.info("Running evaluation on sample conversation")
    logging.info(f"Sample conversation length: {len(sample_conversation)} characters")

    try:
        result = evaluator.evaluate(sample_conversation)

        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Model used: {result.model}")
        print(f"Config file: {result.config_file}")
        print(f"Decode failed: {result.decode_failed}")

        if result.full_response:
            print("\nEvaluation Fields:")
            print("-" * 60)
            for field_name, field_value in result.full_response.items():
                print(f"  {field_name}: {field_value}")
        else:
            print("\nNo evaluation response received")

        print("\n" + "=" * 60)

        # Check if evaluation was successful
        if result.decode_failed:
            logging.warning("Evaluation decode failed - check the response")
            return 1
        else:
            logging.info("Evaluation completed successfully")
            return 0

    except Exception as e:
        logging.exception(f"Error during evaluation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
