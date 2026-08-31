#!/usr/bin/env python3
"""Author and validate the human-gold evaluation corpus.

Every document is synthetic: the people, organizations, identifiers, and
addresses are invented, so the corpus carries no real personal data and lives in
the tracked training/data/gold/ directory. Labels follow the V1 public-sharing
redaction policy (red.pdf). Labeling decisions taken where the policy is open:

- Pronouns: he/him/his/she/her/hers are always labeled for redaction;
  they/them/their/theirs only when they refer to a person; I/me/my/you/your and
  it/its are kept.
- Company tax IDs (INN, GSTIN) are labeled for redaction, matching the existing
  private regression suite, even where no person is attached.
- Cities are redacted inside home/work addresses and as a person's travel
  destination; operational cities (warehouses, branches, hubs) stay.
- Serial and order numbers are redacted (trackable to a purchase); invoice,
  batch, flight, voucher, tracking, and project identifiers are kept.
- The organization that authored a document (its letterhead) stays visible;
  third-party facility mentions are redacted.
- Case/claim numbers tied to a person are redacted; court statute citations are
  kept.
- "morning"/"night"/"evening" are kept except when paired with a clock time and
  a person, where the time expression and the day-part word are redacted.
- ALL email addresses are redacted, role inboxes (hr@, procurement@) included —
  one blunt V1 rule instead of a personal/role judgment call.
- Languages spoken and nationality adjectives ("Dutch", "Indian passport") are
  kept; the policy marks group membership as TBD.

Every redact/keep string is matched case-insensitively on word boundaries, and
ALL of its occurrences in the document count. The builder refuses to emit a
corpus where a labeled string does not occur in its document.
"""
import json
import re
import sys
from pathlib import Path

GOLD_DIR = Path(__file__).resolve().parent / "data" / "gold"

# ---------------------------------------------------------------- dev set (25)
DEV = [
 dict(id="gold_d01_discharge.txt",
  text=("DISCHARGE SUMMARY\nSt. Anthony Riverside Hospital\n"
        "Patient: Marina Kovaleva\nDOB: 03/14/1988\nMRN: patient ID 5527-8841\n"
        "Admitted 02/11/2026, discharged 02/18/2026.\n"
        "Marina Kovaleva was treated for pneumonia. She responded well to the "
        "antibiotics and took her medication every morning without assistance. "
        "The ward nurse recorded stable vitals. Follow-up in three weeks.\n"
        "Discharge approved by the attending physician."),
  redact=[("Marina Kovaleva","PERSON"),("03/14/1988","DOB"),
          ("5527-8841","GENERIC_ID"),("she","PRONOUN"),("her","PRONOUN")],
  keep=["St. Anthony Riverside Hospital","02/11/2026","02/18/2026","morning","ward nurse","pneumonia"]),

 dict(id="gold_d02_hr_letter.txt",
  text=("EMPLOYMENT VERIFICATION\n\nTo whom it may concern,\n"
        "This letter confirms that Grace Whitfield has been employed at Northwind "
        "Logistics LLC as a Senior Accountant since 04/02/2023. Her annual salary "
        "is $84,000. She works full time at our headquarters.\n"
        "For questions call 555-284-3917 or write to hr@northwind-logistics.example.\n"
        "Payroll reference: employee ID NW-2214."),
  redact=[("Grace Whitfield","PERSON"),("Northwind Logistics LLC","EMPLOYER_ORG"),
          ("her","PRONOUN"),("she","PRONOUN"),("555-284-3917","PHONE"),
          ("hr@northwind-logistics.example","EMAIL"),("NW-2214","GENERIC_ID")],
  keep=["Senior Accountant","04/02/2023","84,000","full time"]),

 dict(id="gold_d03_support_chat.txt",
  text=("SUPPORT TRANSCRIPT #48213\n"
        "[10:02] Priya Raman: Hello, thanks for contacting us.\n"
        "[10:03] Mark Osei: Hi, my order 118-4402987 never arrived.\n"
        "[10:04] Priya Raman: I can check that. Could you confirm the customer "
        "number on the account?\n"
        "[10:05] Mark Osei: Sure, customer number 00394471.\n"
        "[10:06] Priya Raman: Found it. The parcel left the Rotterdam warehouse "
        "on Tuesday. He should receive a new tracking link today.\n"
        "[10:07] Mark Osei: Great, thanks."),
  redact=[("Priya Raman","PERSON"),("Mark Osei","PERSON"),
          ("00394471","GENERIC_ID"),("118-4402987","ORDER_NUMBER"),
          ("he","PRONOUN")],
  keep=["Rotterdam","Tuesday","48213","tracking link"]),

 dict(id="gold_d04_email_thread.txt",
  text=("From: tomas.lindqvist@borealfreight.example\n"
        "To: aisha.bello@borealfreight.example\n"
        "Subject: RE: Q3 audit scope\n\n"
        "Aisha, the auditors want the warehouse figures before Friday. Tomas "
        "Lindqvist said he can pull them tonight. Please align the report with "
        "the Meridian Standard 9001 guidelines before it goes out.\n\n"
        "> On Monday Aisha Bello wrote:\n"
        "> The draft is ready. His comments from last week are already merged.\n"),
  redact=[("tomas.lindqvist@borealfreight.example","EMAIL"),
          ("aisha.bello@borealfreight.example","EMAIL"),
          ("Tomas Lindqvist","PERSON"),("Aisha Bello","PERSON"),
          ("he","PRONOUN"),("his","PRONOUN"),("Aisha","PERSON")],
  keep=["Q3 audit scope","Friday","Monday","Meridian Standard 9001"]),

 dict(id="gold_d05_invoice.txt",
  text=("TAX INVOICE No. INV-2026-00871\nDate of issue: 05/06/2026\n"
        "Seller: Vostok Instruments LLC, INN 7734556677\n"
        "Buyer: Leonid Tarasov, 44 Kirovsky Prospekt, apt. 12, Voronezh\n"
        "Item: bench multimeter, qty 3, unit price 12,400.00\n"
        "Subtotal 37,200.00, VAT 7,440.00, total 44,640.00.\n"
        "Payment due within 30 days of the issue date."),
  redact=[("Leonid Tarasov","PERSON"),("7734556677","INN"),
          ("44 Kirovsky Prospekt","ADDRESS"),("apt. 12","ADDRESS"),
          ("Voronezh","CITY_RESIDENCE")],
  keep=["INV-2026-00871","05/06/2026","12,400.00","44,640.00","30 days",
        "bench multimeter"]),

 dict(id="gold_d06_resume.txt",
  text=("RESUME\nDaniil Petrenko\nEmail: d.petrenko@mailbox.example | "
        "Phone: 555-901-2284\nAddress: 17 Maple Crescent, Apt 3B, Springfield, IL 62704\n\n"
        "Education: Riverside State University, B.Sc. Computer Science, 2019.\n"
        "Experience: Software Engineer at Helix Dynamics (2019-2023); he led the "
        "billing team. Backend Developer at Quanta Retail Group since 2023.\n"
        "Skills: Python, SQL, distributed systems."),
  redact=[("Daniil Petrenko","PERSON"),("d.petrenko@mailbox.example","EMAIL"),
          ("555-901-2284","PHONE"),("17 Maple Crescent","ADDRESS"),
          ("Apt 3B","ADDRESS"),("Springfield","CITY_RESIDENCE"),("62704","ZIP"),
          ("Riverside State University","SENSITIVE_ORG"),
          ("Helix Dynamics","EMPLOYER_ORG"),("Quanta Retail Group","EMPLOYER_ORG"),
          ("he","PRONOUN")],
  keep=["Python","SQL","distributed systems","Software Engineer",
        "Backend Developer","2019"]),

 dict(id="gold_d07_school_record.txt",
  text=("ACADEMIC RECORD\nLindenwood Secondary School\n"
        "Student: April Nowak, class 9-B\nStudent ID: LW-2024-1187\n"
        "Enrolled: 09/01/2024. GPA: 4.6 of 5.0.\n"
        "April Nowak represented the school at the regional mathematics olympiad. "
        "Her homeroom teacher notes consistent progress in physics and algebra.\n"
        "Report generated 06/20/2026."),
  redact=[("April Nowak","PERSON"),("9-B","GROUP_CLASS"),
          ("LW-2024-1187","GENERIC_ID"),("her","PRONOUN")],
  keep=["Lindenwood Secondary School","09/01/2024","06/20/2026","4.6",
        "mathematics olympiad","physics","algebra","homeroom teacher"]),

 dict(id="gold_d08_insurance_claim.txt",
  text=("MOTOR CLAIM FORM\nClaim No. CLM-77-58201\n"
        "Policyholder: Viktor Malinov\nPolicy No.: 88 5510 2233 4455 09\n"
        "Vehicle: Skoda Octavia, license plate B 214 CE 77, "
        "VIN TMBJF25L0C6012347.\n"
        "Incident date: 04/28/2026, low-speed collision in a parking lot.\n"
        "Estimated repair cost 61,300.00. He requests payment by bank transfer.\n"
        "Assessor visit scheduled for next Thursday."),
  redact=[("Viktor Malinov","PERSON"),("CLM-77-58201","GENERIC_ID"),
          ("88 5510 2233 4455 09","POLICY_NUMBER"),
          ("B 214 CE 77","LICENSE_PLATE"),("TMBJF25L0C6012347","VIN"),
          ("he","PRONOUN")],
  keep=["04/28/2026","61,300.00","Thursday","parking lot","Skoda Octavia"]),

 dict(id="gold_d09_lease.txt",
  text=("RESIDENTIAL LEASE AGREEMENT\n"
        "Landlord: Sofia Almeida. Tenant: Ruslan Gadzhiev.\n"
        "Premises: 208 Willow Bend Road, Apt 14, Portsmouth.\n"
        "Term: 12 months from 07/01/2026. Monthly rent 1,450.00 payable to "
        "account No. 40817810655009988771 by the fifth business day.\n"
        "The tenant confirmed that she inspected the premises and accepted "
        "their condition. Utilities are billed separately."),
  redact=[("Sofia Almeida","PERSON"),("Ruslan Gadzhiev","PERSON"),
          ("208 Willow Bend Road","ADDRESS"),("Apt 14","ADDRESS"),
          ("Portsmouth","CITY_RESIDENCE"),
          ("40817810655009988771","BANK_ACCOUNT"),("she","PRONOUN")],
  keep=["12 months","07/01/2026","1,450.00","fifth business day","Utilities"]),

 dict(id="gold_d10_bank_letter.txt",
  text=("Dear customer,\n"
        "We confirm that Helena Brandt holds account number 55008812349901 at "
        "our institution. The current balance is 9,315.20. IBAN "
        "DE89370400440532013000 is active for international transfers.\n"
        "She may collect the card at our Frankfurt branch during business "
        "hours. This confirmation is issued on 06/02/2026 for visa purposes "
        "and expires in 90 days."),
  redact=[("Helena Brandt","PERSON"),("55008812349901","GENERIC_ID"),
          ("DE89370400440532013000","IBAN"),("she","PRONOUN")],
  keep=["9,315.20","Frankfurt","06/02/2026","90 days","business hours"]),

 dict(id="gold_d11_incident_report.txt",
  text=("SECURITY INCIDENT REPORT\nSite: eastern loading dock.\n"
        "Guard on duty: Anton Reznik, badge number 4471.\n"
        "Summary: a contractor propped open the service door during the night "
        "shift. He left the building at 2 o'clock in the morning without "
        "signing the register. Cameras recorded the corridor.\n"
        "Anton Reznik filed this report before the end of the shift. "
        "Maintenance replaced the door closer the same day."),
  redact=[("Anton Reznik","PERSON"),("4471","GENERIC_ID"),("he","PRONOUN"),
          ("2 o'clock","TIME"),("morning","TIME")],
  keep=["night","eastern loading dock","service door","door closer"]),

 dict(id="gold_d12_minutes.txt",
  text=("PROJECT MEETING MINUTES — project code ORION-7\n"
        "Attendees: Olga Sereda, Jonas Keller.\n"
        "Budget line approved at 240,000.00 for the second phase.\n"
        "Olga Sereda presented the vendor comparison; they agreed to shortlist "
        "two suppliers. Jonas Keller will draft the contract by 07/15/2026.\n"
        "Next meeting in two weeks in the large conference room."),
  redact=[("Olga Sereda","PERSON"),("Jonas Keller","PERSON"),
          ("they","PRONOUN")],
  keep=["ORION-7","240,000.00","07/15/2026","two suppliers","conference room"]),

 dict(id="gold_d13_court_filing.txt",
  text=("STATEMENT OF CLAIM\nCase No. 2-1482/2026\n"
        "Plaintiff: Zoya Ignatova, born 07/22/1990, passport series 45 19 "
        "No. 882201, residing at 6 Gagarina Street, apt. 40, Tula.\n"
        "Defendant: Stroyservice LLC.\n"
        "Under Article 309 of the Civil Code the plaintiff demands performance "
        "of the contract dated 03/12/2025. She paid 78,500.00 in advance.\n"
        "Attachments: contract copy, payment receipts."),
  redact=[("Zoya Ignatova","PERSON"),("07/22/1990","DOB"),
          ("45 19 No. 882201","PASSPORT"),("2-1482/2026","GENERIC_ID"),
          ("6 Gagarina Street","ADDRESS"),("apt. 40","ADDRESS"),
          ("Tula","CITY_RESIDENCE"),("she","PRONOUN")],
  keep=["Article 309 of the Civil Code","03/12/2025","78,500.00",
        "payment receipts","Stroyservice LLC"]),

 dict(id="gold_d14_visa_form.txt",
  text=("VISA APPLICATION SUMMARY\n"
        "Applicant: Bruno Carvalho, born on 5 May 1994.\n"
        "Passport No. X8250147, valid until 11/30/2031.\n"
        "Purpose of travel: academic conference, five days.\n"
        "The applicant covers all expenses himself; estimated budget 2,100.00. "
        "Contact email: b.carvalho@postbox.example.\n"
        "Supporting documents: invitation letter, hotel booking."),
  redact=[("Bruno Carvalho","PERSON"),("5 May 1994","DOB"),
          ("X8250147","PASSPORT"),("b.carvalho@postbox.example","EMAIL")],
  keep=["11/30/2031","academic conference","five days","2,100.00",
        "invitation letter","hotel booking"]),

 dict(id="gold_d15_police_report.txt",
  text=("TRAFFIC INCIDENT RECORD\n"
        "Witness: Frank Moreau, contacted at 555-702-6635.\n"
        "A grey hatchback, registration AB12 CDE, clipped a parked van on the "
        "Leeds ring road and did not stop. The witness resides at 31 Alder "
        "Grove, Leicester, and agreed to provide a statement.\n"
        "Traffic on the ring road was heavy but moving. No injuries were "
        "reported. Officers reviewed the roadside footage the same evening."),
  redact=[("Frank Moreau","PERSON"),("555-702-6635","PHONE"),
          ("AB12 CDE","LICENSE_PLATE"),("31 Alder Grove","ADDRESS"),
          ("Leicester","CITY_RESIDENCE")],
  keep=["Leeds","grey hatchback","No injuries","evening","roadside footage"]),

 dict(id="gold_d16_clinic_intake.txt",
  text=("NEW PATIENT INTAKE\nWillow Creek Community Clinic\n"
        "Patient: Nadia Fetisova\nDate of birth: 09/02/1985\n"
        "Insurance policy No.: 61 7788 9900 1122 33\n"
        "Emergency contact: Pavel Fetisov, 555-448-2160.\n"
        "She reports seasonal allergies and takes no regular medication. "
        "Preferred appointment time is late afternoon.\n"
        "Intake completed 06/10/2026 by front-desk staff."),
  redact=[("Nadia Fetisova","PERSON"),("09/02/1985","DOB"),
          ("61 7788 9900 1122 33","POLICY_NUMBER"),
          ("Pavel Fetisov","PERSON"),("555-448-2160","PHONE"),
          ("she","PRONOUN")],
  keep=["Willow Creek Community Clinic","06/10/2026","seasonal allergies",
        "late afternoon","front-desk staff"]),

 dict(id="gold_d17_employment_contract.txt",
  text=("EMPLOYMENT CONTRACT No. 44/26\n"
        "Employer: Sibir Composites LLC, INN 5406778899.\n"
        "Employee: Kirill Vetrov, SNILS 214-556-778 90.\n"
        "Position: process engineer. Monthly salary 96,000.00 before tax. "
        "Probation period: three months.\n"
        "The employee starts on 08/03/2026. He reports to the head of the "
        "composites workshop. Working hours are set by the internal labor "
        "regulations."),
  redact=[("Sibir Composites LLC","EMPLOYER_ORG"),("5406778899","INN"),
          ("Kirill Vetrov","PERSON"),("214-556-778 90","SNILS"),
          ("he","PRONOUN")],
  keep=["process engineer","96,000.00","three months","08/03/2026",
        "44/26","internal labor regulations"]),

 dict(id="gold_d18_delivery_note.txt",
  text=("DELIVERY NOTE\nShipment tracking number 1Z-884-220-19.\n"
        "Contents: 24 boxes of ceramic tile, gross weight 618 kg.\n"
        "Dispatched from the Katowice warehouse; the truck takes the night "
        "route to avoid daytime congestion. Expected transit time is 31 hours.\n"
        "Pallet count verified twice at the gate. Reference batch 7742118850 "
        "printed on every box label. No signature required at drop-off."),
  redact=[],
  keep=["1Z-884-220-19","24 boxes","618 kg","Katowice","night",
        "31 hours","7742118850"]),

 dict(id="gold_d19_press_release.txt",
  text=("PRESS RELEASE\nMeridian Foods opens a new distribution hub.\n"
        "The company invested 4,500,000.00 in the facility and expects to "
        "ship 60,000 pallets a year. The hub is located next to the Gdansk "
        "port terminal and will serve retail chains across the region.\n"
        "Recruitment for 120 warehouse roles starts in the autumn. Applications "
        "open through the public careers portal."),
  redact=[],
  keep=["Meridian Foods","4,500,000.00","60,000 pallets","Gdansk",
        "120 warehouse roles","careers portal"]),

 dict(id="gold_d20_utility_bill.txt",
  text=("ELECTRICITY BILL — June 2026\n"
        "Account holder: Tamara Yusupova\n"
        "Supply address: 90 Chestnut Field Road, apt. 7, Samara\n"
        "Personal account No.: 5501-XN-887702\n"
        "Meter reading previous 04812, current 05147, consumed 335 kWh.\n"
        "Tariff 6.42 per kWh, amount due 2,150.70 by 07/25/2026.\n"
        "Payments made after the due date incur a late fee."),
  redact=[("Tamara Yusupova","PERSON"),("90 Chestnut Field Road","ADDRESS"),
          ("apt. 7","ADDRESS"),("Samara","CITY_RESIDENCE"),
          ("5501-XN-887702","GENERIC_ID")],
  keep=["04812","05147","335 kWh","6.42","2,150.70","07/25/2026","late fee"]),

 dict(id="gold_d21_payroll_memo.txt",
  text=("PAYROLL MEMO — internal\n"
        "New hire setup for Denis Okafor is complete. SSN 536-90-4271 was "
        "entered into the system, and direct deposit goes to account number "
        "77120095534482 starting with the July cycle.\n"
        "His gross monthly pay is 7,300.00 with standard withholding. The "
        "finance team keeps original forms for seven years per the retention "
        "schedule."),
  redact=[("Denis Okafor","PERSON"),("536-90-4271","SSN"),
          ("77120095534482","GENERIC_ID"),("his","PRONOUN")],
  keep=["July cycle","7,300.00","seven years","retention schedule"]),

 dict(id="gold_d22_itinerary.txt",
  text=("TRAVEL ITINERARY\nTraveler: Yana Melnyk\n"
        "Passport No. P0442158 must match the booking exactly.\n"
        "Outbound flight LH1407 departs 09/12/2026 at 07:40, arrival Lisbon. "
        "Return flight LH1408 on 09/19/2026. Hotel voucher HTL-58821 covers "
        "seven nights.\n"
        "Baggage allowance one checked bag, 23 kg. Check-in opens 48 hours "
        "before departure."),
  redact=[("Yana Melnyk","PERSON"),("P0442158","PASSPORT"),
          ("Lisbon","CITY_TRAVEL")],
  keep=["LH1407","LH1408","09/12/2026","09/19/2026","HTL-58821",
        "23 kg","48 hours"]),

 dict(id="gold_d23_shelter_intake.txt",
  text=("CONFIDENTIAL INTAKE SHEET\nHaven House Women's Shelter\n"
        "Resident: Larisa Bondar, case ID HH-2026-031.\n"
        "Caseworker: Miriam Adler.\n"
        "She arrived with documents and requested help with housing "
        "applications. Her belongings were logged and stored. Safety planning "
        "session scheduled within 72 hours of arrival.\n"
        "Access to this sheet is restricted to assigned staff."),
  redact=[("Larisa Bondar","PERSON"),("HH-2026-031","GENERIC_ID"),
          ("Miriam Adler","PERSON"),("she","PRONOUN"),("her","PRONOUN")],
  keep=["Haven House Women's Shelter","72 hours","housing applications",
        "Safety planning","assigned staff"]),

 dict(id="gold_d24_warranty_claim.txt",
  text=("WARRANTY CLAIM\nProduct: cordless drill, serial number DRL8802914476.\n"
        "Purchased 03/18/2026 at an authorized dealer, receipt retained.\n"
        "Customer: Igor Stachowiak, phone 555-330-8174, email "
        "i.stachowiak@inbox.example.\n"
        "Reported fault: battery does not hold charge after ten cycles. "
        "Replacement part ships within five business days once the claim is "
        "approved by the service center."),
  redact=[("Igor Stachowiak","PERSON"),("555-330-8174","PHONE"),
          ("i.stachowiak@inbox.example","EMAIL"),
          ("DRL8802914476","SERIAL_NUMBER")],
  keep=["03/18/2026","ten cycles","five business days",
        "service center","cordless drill"]),

 dict(id="gold_d25_job_offer.txt",
  text=("OFFER OF EMPLOYMENT\n"
        "Dear Alina Sorokina,\n"
        "Kestrel Analytics is pleased to offer you the position of Data "
        "Analyst starting 09/01/2026 with an annual salary of 71,000.00.\n"
        "You will report to Robert Ancelet at our office at 12 Foundry Lane, "
        "Manchester. Please confirm your acceptance within ten business days.\n"
        "This offer is contingent on standard reference checks."),
  redact=[("Alina Sorokina","PERSON"),("Kestrel Analytics","EMPLOYER_ORG"),
          ("Robert Ancelet","PERSON"),("12 Foundry Lane","ADDRESS"),
          ("Manchester","CITY_RESIDENCE")],
  keep=["Data Analyst","09/01/2026","71,000.00","ten business days",
        "reference checks"]),
]

# --------------------------------------------------------------- test set (15)
TEST = [
 dict(id="gold_t01_discharge.txt",
  text=("DISCHARGE NOTE\nBlue Hills Municipal Hospital\n"
        "Patient: Bill Tanaka, DOB: 22.07.1990, MRN patient ID 8814-0223.\n"
        "Bill Tanaka was admitted on 05/03/2026 with a fractured wrist and "
        "discharged on 05/05/2026. He tolerated the cast well and walked the "
        "corridor every evening with supervision.\n"
        "Physiotherapy referral issued; first session in ten days."),
  redact=[("Bill Tanaka","PERSON"),("22.07.1990","DOB"),
          ("8814-0223","GENERIC_ID"),("he","PRONOUN")],
  keep=["Blue Hills Municipal Hospital","05/03/2026","05/05/2026","evening",
        "Physiotherapy","ten days"]),

 dict(id="gold_t02_hr_letter.txt",
  text=("TO WHOM IT MAY CONCERN\n"
        "Rose Adeyemi has worked at Calder Marine Services since 02/10/2021 "
        "as a Logistics Coordinator. Her employment is full time and her "
        "monthly salary is 5,900.00.\n"
        "Employee ID CM-0977 is assigned for internal records. Questions may "
        "be directed to 555-612-4903 or people@caldermarine.example.\n"
        "Issued on 06/28/2026 at the request of the employee."),
  redact=[("Rose Adeyemi","PERSON"),("Calder Marine Services","EMPLOYER_ORG"),
          ("her","PRONOUN"),("CM-0977","GENERIC_ID"),
          ("555-612-4903","PHONE"),("people@caldermarine.example","EMAIL")],
  keep=["Logistics Coordinator","02/10/2021","5,900.00","06/28/2026",
        "full time"]),

 dict(id="gold_t03_support_chat.txt",
  text=("CHAT LOG #90417\n"
        "[14:11] Support: Good afternoon, how can we help?\n"
        "[14:12] Customer: My name is Petr Havel, the dishwasher under order "
        "552-9083311 leaks.\n"
        "[14:13] Support: Sorry to hear that. Can you give the member number "
        "from the loyalty card?\n"
        "[14:14] Petr Havel: member number 7745120.\n"
        "[14:15] Support: Thank you. A technician will call him tomorrow "
        "between nine and noon."),
  redact=[("Petr Havel","PERSON"),("7745120","GENERIC_ID"),
          ("552-9083311","ORDER_NUMBER"),("him","PRONOUN")],
  keep=["90417","dishwasher","tomorrow","loyalty card"]),

 dict(id="gold_t04_email.txt",
  text=("From: monika.szabo@ferrumworks.example\n"
        "To: procurement@ferrumworks.example\n"
        "Subject: replacement rollers\n\n"
        "Team, the supplier confirmed the shipment for Wednesday. Monika "
        "Szabo asked that the invoice be checked against the Corvus Framework "
        "checklist before payment. She flagged two duplicate lines last "
        "quarter, so please compare totals carefully.\n"
        "Reference the shared folder for the spec sheets."),
  redact=[("monika.szabo@ferrumworks.example","EMAIL"),
          ("procurement@ferrumworks.example","EMAIL"),
          ("Monika Szabo","PERSON"),("she","PRONOUN")],
  keep=["Wednesday","Corvus Framework","two duplicate lines","spec sheets"]),

 dict(id="gold_t05_invoice.txt",
  text=("INVOICE No. 2026/0455\nIssued: 06/15/2026\n"
        "Supplier: Argon Fittings LLC, INN 6659887701\n"
        "Customer: Olesya Vinnik, 5 Bramble Court, Apt 2, Norwich\n"
        "Goods: pipe couplings, 140 units at 86.50 each.\n"
        "Total 12,110.00 including VAT. Late payments accrue 0.1 percent "
        "per day. Goods remain supplier property until paid in full."),
  redact=[("Olesya Vinnik","PERSON"),("6659887701","INN"),
          ("5 Bramble Court","ADDRESS"),("Apt 2","ADDRESS"),
          ("Norwich","CITY_RESIDENCE")],
  keep=["2026/0455","06/15/2026","140 units","86.50","12,110.00",
        "0.1 percent","pipe couplings"]),

 dict(id="gold_t06_cv.txt",
  text=("CURRICULUM VITAE\nCandidate: Anouk Devos\n"
        "Contact: a.devos@courrier.example, 555-118-6420.\n"
        "Home address: 63 Heron Walk, Apt 11, Bristol, BS1 4ND.\n"
        "Education: Millbrook Technical College, diploma in mechatronics.\n"
        "Employment: maintenance planner at Trask Elevators since 2020; she "
        "schedules inspections across the southern region.\n"
        "Languages: Dutch, English, French."),
  redact=[("Anouk Devos","PERSON"),("a.devos@courrier.example","EMAIL"),
          ("555-118-6420","PHONE"),("63 Heron Walk","ADDRESS"),
          ("Apt 11","ADDRESS"),("Bristol","CITY_RESIDENCE"),("BS1 4ND","ZIP"),
          ("Millbrook Technical College","SENSITIVE_ORG"),
          ("Trask Elevators","EMPLOYER_ORG"),("she","PRONOUN")],
  keep=["mechatronics","maintenance planner","Dutch","English","French",
        "southern region"]),

 dict(id="gold_t07_school_letter.txt",
  text=("PARENT NOTIFICATION\nOakfield Grammar School\n"
        "Student: Tomas Vilkas, grade 7-A, student ID OG-3391-77.\n"
        "Dear parent, the autumn term begins 09/02/2026. Tomas Vilkas was "
        "selected for the debate team; travel consent forms are due by "
        "08/20/2026. His locker assignment stays the same.\n"
        "The school office is open on weekdays."),
  redact=[("Tomas Vilkas","PERSON"),("7-A","GROUP_CLASS"),
          ("OG-3391-77","GENERIC_ID"),("his","PRONOUN")],
  keep=["Oakfield Grammar School","09/02/2026","08/20/2026","debate team",
        "weekdays","autumn term"]),

 dict(id="gold_t08_claim.txt",
  text=("PROPERTY CLAIM\nClaim reference CLM-2026-8817 opened for water "
        "damage.\nClaimant: Sergei Ustinov, policy No.: 74 2211 5566 8899 04.\n"
        "Damaged rooms: kitchen and hallway, estimate 9,850.00.\n"
        "An adjuster inspects the flat on 07/08/2026 in the afternoon. He "
        "photographed the ceiling on the first visit. Repairs may start only "
        "after written approval."),
  redact=[("Sergei Ustinov","PERSON"),("CLM-2026-8817","GENERIC_ID"),
          ("74 2211 5566 8899 04","POLICY_NUMBER"),("he","PRONOUN")],
  keep=["9,850.00","07/08/2026","afternoon","kitchen and hallway",
        "written approval"]),

 dict(id="gold_t09_lease.txt",
  text=("SHORT-TERM LEASE\n"
        "Owner: Marta Kowalczyk. Renter: Eldar Aliyev.\n"
        "Unit: 402 Juniper House, 15 Carding Street, Dundee.\n"
        "Rent 980.00 per month, deposit equal to one month.\n"
        "Payments go to account No. 40817810233445566778. The renter said he "
        "will register the parking permit separately. Smoking is not allowed "
        "anywhere in the building."),
  redact=[("Marta Kowalczyk","PERSON"),("Eldar Aliyev","PERSON"),
          ("402 Juniper House","ADDRESS"),("15 Carding Street","ADDRESS"),
          ("Dundee","CITY_RESIDENCE"),
          ("40817810233445566778","BANK_ACCOUNT"),("he","PRONOUN")],
  keep=["980.00","one month","parking permit","Smoking"]),

 dict(id="gold_t10_bank_notice.txt",
  text=("NOTICE OF ACCOUNT CHANGE\n"
        "Customer: Ivo Marinov. The card linked to account number "
        "88770045512099 will be reissued; the old card stops working on "
        "08/01/2026.\n"
        "National insurance number CE929255B was verified during the review, "
        "and his contact details were confirmed unchanged.\n"
        "Standard fees apply per the published tariff schedule. Visit any "
        "branch with photo identification to collect the new card."),
  redact=[("Ivo Marinov","PERSON"),("88770045512099","GENERIC_ID"),
          ("CE929255B","UK_NINO"),("his","PRONOUN")],
  keep=["08/01/2026","tariff schedule","photo identification"]),

 dict(id="gold_t11_incident.txt",
  text=("WAREHOUSE INCIDENT LOG\n"
        "Reported by: Dmytro Savchenko, badge number 2209.\n"
        "A forklift clipped rack 14 during the evening shift. She stopped the "
        "vehicle at 9 o'clock in the evening and closed the aisle before "
        "anyone was hurt. Photographs were attached to the safety file.\n"
        "Rack inspection passed the next day; no structural damage found. "
        "The operator completed a refresher course."),
  redact=[("Dmytro Savchenko","PERSON"),("2209","GENERIC_ID"),
          ("she","PRONOUN"),("9 o'clock in the evening","TIME")],
  keep=["evening shift","rack 14","forklift","refresher course",
        "no structural damage"]),

 dict(id="gold_t12_minutes.txt",
  text=("BOARD MINUTES — initiative VEGA-3\n"
        "Present: Camille Fournier, Andrei Rusu.\n"
        "The quarterly spend of 310,000.00 was ratified. Camille Fournier "
        "presented the risk register; they asked for a follow-up audit of the "
        "logistics contract before 08/30/2026.\n"
        "Minutes circulated to the distribution list the same week."),
  redact=[("Camille Fournier","PERSON"),("Andrei Rusu","PERSON"),
          ("they","PRONOUN")],
  keep=["VEGA-3","310,000.00","08/30/2026","risk register",
        "distribution list"]),

 dict(id="gold_t13_court_notice.txt",
  text=("COURT NOTICE\nCase No. 5-0912/2026\n"
        "Respondent: Aigerim Bekova, born on March 3, 1985, residing at 27 "
        "Falcon Rise, apt. 9, Aberdeen.\n"
        "Hearing scheduled for 09/14/2026 under Article 154 of the Civil "
        "Procedure Code. She must bring originals of the payment documents.\n"
        "Failure to appear may result in a default judgment."),
  redact=[("Aigerim Bekova","PERSON"),("March 3, 1985","DOB"),
          ("5-0912/2026","GENERIC_ID"),("27 Falcon Rise","ADDRESS"),
          ("apt. 9","ADDRESS"),("Aberdeen","CITY_RESIDENCE"),
          ("she","PRONOUN")],
  keep=["09/14/2026","Article 154 of the Civil Procedure Code",
        "default judgment","originals"]),

 dict(id="gold_t14_bulletin.txt",
  text=("FACILITY BULLETIN\n"
        "The cafeteria switches to the summer menu on Monday. Elevator "
        "maintenance runs overnight from 23:00 to 05:00; use the west "
        "staircase during the night window.\n"
        "The annual fire drill is scheduled for the last Friday of the "
        "month. Badge readers at the north entrance were upgraded and now "
        "respond faster. Direct questions to the facilities desk."),
  redact=[],
  keep=["summer menu","Monday","23:00","05:00","night","fire drill",
        "north entrance","facilities desk"]),

 dict(id="gold_t15_intake.txt",
  text=("COUNSELING INTAKE SUMMARY\nHarbor Light Family Support Center\n"
        "Client: Beatrix Molnar, date of birth 12/08/1979.\n"
        "Contact phone 555-953-7788, client record HL-455-2026.\n"
        "She asked for an interpreter for the first session and prefers "
        "appointments on weekday mornings. Her consent forms were signed and "
        "filed the same day.\n"
        "Next review after four sessions."),
  redact=[("Beatrix Molnar","PERSON"),("12/08/1979","DOB"),
          ("555-953-7788","PHONE"),("HL-455-2026","GENERIC_ID"),
          ("she","PRONOUN"),("her","PRONOUN")],
  keep=["Harbor Light Family Support Center","interpreter","weekday mornings",
        "four sessions","consent forms"]),
]


# ----------------------------------------------- fresh held-out batch (10)
# Authored after the rules phase, never used for tuning. Replaces the original
# test set as the clean generalization measurement (that one absorbed five
# fixes from its first run and is now semi-dev).
TEST2 = [
 dict(id="gold_v2_01_vet_invoice.txt",
  text=("VETERINARY INVOICE No. VET-2026-118\n"
        "Owner: Adaeze Okonkwo, 21 Foxglove Walk, Apt 5, Exeter.\n"
        "Patient animal: Biscuit, domestic shorthair, 4 years.\n"
        "Procedure: dental cleaning, 180.00. Follow-up on 09/22/2026.\n"
        "She asked for the vaccination record to be emailed to "
        "a.okonkwo@postfach.example."),
  redact=[("Adaeze Okonkwo","PERSON"),("21 Foxglove Walk","ADDRESS"),
          ("Apt 5","ADDRESS"),("Exeter","CITY_RESIDENCE"),("she","PRONOUN"),
          ("a.okonkwo@postfach.example","EMAIL")],
  keep=["VET-2026-118","Biscuit","180.00","09/22/2026","dental cleaning",
        "4 years"]),

 dict(id="gold_v2_02_reference_letter.txt",
  text=("REFERENCE LETTER\n"
        "June Callahan worked at Bramford Print Works from 2018 to 2024 as a "
        "Production Supervisor. Her attention to detail was excellent, and "
        "she trained twelve new operators over six years.\n"
        "Contact me on 555-207-8841 with any questions.\n"
        "Issued 07/30/2026 at the request of the employee."),
  redact=[("June Callahan","PERSON"),("Bramford Print Works","EMPLOYER_ORG"),
          ("her","PRONOUN"),("she","PRONOUN"),("555-207-8841","PHONE")],
  keep=["Production Supervisor","twelve new operators","07/30/2026",
        "six years"]),

 dict(id="gold_v2_03_appointment.txt",
  text=("APPOINTMENT REMINDER\nEastbrook Dental Clinic\n"
        "Patient: Milan Horvat, born on the 2nd of June, 1990.\n"
        "Your cleaning is booked for 10/05/2026 at 14:30. Arrive ten minutes "
        "early. He should bring the insurance card and a list of current "
        "medications.\n"
        "To reschedule, reply to this message before Friday."),
  redact=[("Milan Horvat","PERSON"),("2nd of June, 1990","DOB"),
          ("he","PRONOUN")],
  keep=["Eastbrook Dental Clinic","10/05/2026","14:30","ten minutes",
        "Friday","insurance card"]),

 dict(id="gold_v2_04_gym_cancellation.txt",
  text=("MEMBERSHIP CANCELLATION\n"
        "Member: Rustam Aliyev, member number 0077-4413.\n"
        "The direct debit ends after the September payment of 39.90. Access "
        "cards deactivate on the last day of the notice period.\n"
        "His locker must be emptied by 09/30/2026. Personal items left after "
        "that date go to the front desk for thirty days."),
  redact=[("Rustam Aliyev","PERSON"),("0077-4413","GENERIC_ID"),
          ("his","PRONOUN")],
  keep=["39.90","09/30/2026","thirty days","front desk","September"]),

 dict(id="gold_v2_05_tax_letter.txt",
  text=("TAX ASSESSMENT NOTICE\n"
        "Taxpayer: Ingrid Vestergaard\n"
        "Employee no. 8873 of Solvang Furniture ApS per the employer filing. "
        "Assessed amount 4,120.00, payable by 11/01/2026.\n"
        "She may appeal within 30 days under section 42 of the Tax "
        "Administration Act. Late payment adds interest at 0.6 percent per "
        "month."),
  redact=[("Ingrid Vestergaard","PERSON"),("8873","GENERIC_ID"),
          ("Solvang Furniture ApS","EMPLOYER_ORG"),("she","PRONOUN")],
  keep=["4,120.00","11/01/2026","30 days","section 42 of the Tax "
        "Administration Act","0.6 percent"]),

 dict(id="gold_v2_06_small_claims.txt",
  text=("SMALL CLAIMS FILING\nCase No. SC-2026-0331\n"
        "Claimant: Bogdan Lupu, residing at 8 Miller's Row, Colchester.\n"
        "Claim: 950.00 for an unreturned deposit under the tenancy that ended "
        "03/31/2026. He filed the claim online and paid the 35.00 fee.\n"
        "The respondent has 14 days to reply after service."),
  redact=[("Bogdan Lupu","PERSON"),("SC-2026-0331","GENERIC_ID"),
          ("8 Miller's Row","ADDRESS"),("Colchester","CITY_RESIDENCE"),
          ("he","PRONOUN")],
  keep=["950.00","03/31/2026","35.00","14 days"]),

 dict(id="gold_v2_07_onboarding.txt",
  text=("ONBOARDING CHECKLIST\n"
        "New hire: Priyanka Deshmukh, badge number 5501.\n"
        "PAN GHKPD3821L was collected for payroll registration.\n"
        "Laptop issued, asset tag AST-90221. Building tour booked with the "
        "facilities team for Monday morning. Her desk is on the third floor.\n"
        "Probation review after three months."),
  redact=[("Priyanka Deshmukh","PERSON"),("5501","GENERIC_ID"),
          ("GHKPD3821L","IN_PAN"),("her","PRONOUN")],
  keep=["AST-90221","Monday","morning","third floor","three months"]),

 dict(id="gold_v2_08_helpdesk.txt",
  text=("HELPDESK TICKET #77120\n"
        "[09:14] Chinwe Obi: My laptop will not join the office network.\n"
        "[09:16] Support: Thanks. Is the VPN client running?\n"
        "[09:17] Chinwe Obi: Yes, version 4.2.1. Same error since Tuesday.\n"
        "[09:20] Support: We will push a certificate update to her machine "
        "within the hour and call 555-892-3306 if it fails."),
  redact=[("Chinwe Obi","PERSON"),("her","PRONOUN"),
          ("555-892-3306","PHONE")],
  keep=["77120","4.2.1","Tuesday","VPN client","certificate update"]),

 dict(id="gold_v2_09_hotel_booking.txt",
  text=("BOOKING CONFIRMATION QRT-55302\n"
        "Guest: Lauri Nieminen. Two nights, 10/11/2026 to 10/13/2026, twin "
        "room with breakfast, total 214.00.\n"
        "The hotel is a short walk from the Tampere central station. Check-in "
        "from 15:00. He requested a quiet room away from the elevator.\n"
        "Free cancellation until 48 hours before arrival."),
  redact=[("Lauri Nieminen","PERSON"),("he","PRONOUN")],
  keep=["QRT-55302","10/11/2026","10/13/2026","214.00","Tampere","15:00",
        "48 hours","twin room"]),

 dict(id="gold_v2_10_newsletter.txt",
  text=("COMMUNITY NEWSLETTER — October\n"
        "The repair cafe returns to the main hall on the first Saturday of "
        "the month; bring small appliances and bicycles. Volunteers fixed 214 "
        "items last season.\n"
        "The evening yoga group moves to 19:00 for the winter. Parking "
        "permits for the season cost 12.00 and are sold at the entrance.\n"
        "Suggestions go in the box by the noticeboard."),
  redact=[],
  keep=["October","first Saturday","214 items","evening","19:00","12.00",
        "noticeboard"]),
]


def occurrences(needle, haystack):
    pattern = re.escape(needle)
    if needle[0].isalnum():
        pattern = r"\b" + pattern
    if needle[-1].isalnum():
        pattern = pattern + r"\b"
    return [m.span() for m in re.finditer(pattern, haystack, re.IGNORECASE)]


def validate(docs, name):
    errors = []
    for doc in docs:
        for text, _ in doc["redact"]:
            if not occurrences(text, doc["text"]):
                errors.append(f"{name}/{doc['id']}: redact {text!r} not found")
        for text in doc["keep"]:
            if not occurrences(text, doc["text"]):
                errors.append(f"{name}/{doc['id']}: keep {text!r} not found")
        redact_texts = [t.lower() for t, _ in doc["redact"]]
        if len(set(redact_texts)) != len(redact_texts):
            errors.append(f"{name}/{doc['id']}: duplicate redact entries")
    return errors


def emit(docs, docs_path, cases_path):
    with open(docs_path, "w") as f:
        for doc in docs:
            f.write(json.dumps({
                "source_file": doc["id"], "text": doc["text"],
                "spans": [{"text": t, "label": l} for t, l in doc["redact"]],
            }, ensure_ascii=False) + "\n")
    with open(cases_path, "w") as f:
        for doc in docs:
            f.write(json.dumps({
                "source_file": doc["id"],
                "must_redact": [{"text": t, "label": l} for t, l in doc["redact"]],
                "must_keep": [{"text": t} for t in doc["keep"]],
            }, ensure_ascii=False) + "\n")


def main():
    problems = (validate(DEV, "dev") + validate(TEST, "test")
                + validate(TEST2, "test2"))
    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} validation errors — corpus not written")
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    emit(DEV, GOLD_DIR / "dev_docs.jsonl", GOLD_DIR / "dev_cases.jsonl")
    emit(TEST, GOLD_DIR / "test_docs.jsonl", GOLD_DIR / "test_cases.jsonl")
    emit(TEST2, GOLD_DIR / "test2_docs.jsonl", GOLD_DIR / "test2_cases.jsonl")
    n_redact = sum(len(d["redact"]) for d in DEV + TEST + TEST2)
    n_keep = sum(len(d["keep"]) for d in DEV + TEST + TEST2)
    print(f"wrote {len(DEV)} dev + {len(TEST)} test + {len(TEST2)} test2 docs "
          f"({n_redact} redact spans, {n_keep} keep spans) to {GOLD_DIR}")


if __name__ == "__main__":
    main()
