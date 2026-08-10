"""
Create sample text files with PII for testing
"""
import os
from config import DATA_INPUT


def create_sample_text_files():
    """Create sample text files for testing"""
    text_dir = os.path.join(DATA_INPUT, "text_files")
    os.makedirs(text_dir, exist_ok=True)

    sample_files = {
        'email.txt': """
Subject: Meeting Confirmation
From: john.smith@company.com
To: sarah.johnson@partner.org

Dear Sarah,

I hope this email finds you well. I wanted to confirm our meeting scheduled for tomorrow at 2:00 PM.

Please bring the contract documents and your ID for verification. If you need to reach me, you can call me at 555-123-4567 or email me directly.

My office address is 123 Business Ave, Suite 500, New York, NY 10001.

For any emergencies, please contact our main office at (555) 987-6543.

Best regards,
John Smith
Senior Account Manager
john.smith@company.com
Direct: 555-123-4567
""",

        'story.txt': """
The Detective's Case

Detective Maria Rodriguez had been working on the Johnson case for weeks. The victim, David Thompson, lived at 456 Oak Street in downtown Portland. His wife, Jennifer Thompson, had called 911 on the night of March 15th, 2024.

"He never came home from work," Jennifer had told the operator, tears in her voice. "His phone goes straight to voicemail at 503-555-0199."

Maria reviewed her notes. David worked at TechCorp Industries, where his supervisor, Michael Chen, last saw him leaving the office at 6:30 PM. Security footage showed David getting into his blue Honda Civic, license plate ABC-1234.

The break in the case came when a witness, elderly Mrs. Patterson from 789 Pine Lane, called the tip line at 503-555-TIPS. She had seen a man matching David's description at the riverside park around 8:00 PM that night.

Maria's investigation revealed that David had been receiving threatening emails from an anonymous account: threats123@tempmail.com. The last message, sent on March 14th, read: "Pay the $50,000 or face consequences."

Bank records showed David's account (routing number 123456789, account 987654321) had been depleted just days before his disappearance.
""",

        'customer_support.txt': """
CUSTOMER SUPPORT LOG
Date: January 15, 2024

TICKET #CS-2024-001
Customer: Robert Williams
Email: r.williams@email.net
Phone: +1-555-777-8888
Issue: Account access problem
Resolution: Reset password, verified identity using SSN: 123-45-6789
Agent: Lisa Chen

TICKET #CS-2024-002
Customer: Amanda Davis
Email: amanda.davis@workplace.com
Phone: (555) 444-3333
Address: 321 Elm Street, Springfield, IL 62701
Issue: Billing inquiry
Credit Card: **** **** **** 4567 (last 4 digits)
Resolution: Refund processed
Agent: Michael Johnson

TICKET #CS-2024-003
Customer: James Wilson
DOB: 08/22/1985
Email: james.w@personal.org
Phone: 555.222.1111
Issue: Technical support
IP Address: 192.168.1.100 (customer's network)
Resolution: Software update provided
Agent: Sarah Martinez
""",

        'medical_notes.json': """{
  "patient_records": [
    {
      "patient_id": "P001",
      "name": "Elizabeth Johnson",
      "dob": "03/15/1978",
      "ssn": "987-65-4321",
      "phone": "555-888-7777",
      "email": "e.johnson@email.com",
      "address": "789 Health Street, Medical City, CA 90210",
      "insurance_id": "INS123456789",
      "notes": "Patient complained of headaches. Prescribed medication. Follow-up in 2 weeks.",
      "physician": "Dr. Sarah Thompson",
      "physician_phone": "555-MED-CARE"
    },
    {
      "patient_id": "P002",
      "name": "Michael Brown",
      "dob": "11/22/1965",
      "ssn": "456-78-9123",
      "emergency_contact": "Wife: Linda Brown at 555-999-0000",
      "notes": "Regular checkup completed. All vitals normal. Next appointment scheduled."
    }
  ],
  "clinic_info": {
    "name": "Downtown Medical Center",
    "address": "100 Medical Plaza, Health City, CA 90210",
    "phone": "555-MEDICAL",
    "fax": "555-MED-FAX1"
  }
}"""
    }

    created_files = []
    for filename, content in sample_files.items():
        filepath = os.path.join(text_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        created_files.append(filepath)
        print(f"✅ Created: {filename}")

    print(f"\n🎉 Created {len(created_files)} sample text files in {text_dir}")
    return created_files


if __name__ == "__main__":
    print("📝 Creating sample text files with PII...")
    create_sample_text_files()
    print("\n🚀 Ready to test text PII detection!")
    print("   Run: python text_pii_detector.py")
