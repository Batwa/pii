#!/usr/bin/env python3
"""Author and validate the human-gold evaluation corpus.

Every document is synthetic: the people, organizations, identifiers, and
addresses are invented (the 25 dev / 15 test / 10 test2 / 13 test3 / 13 test4 / 13 test5 / 13 test6 / 13 test7 / 13 test8 documents
are listed below in authoring order), so the corpus carries no real personal data and lives in
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
- A bank sort code is labeled with the account number it accompanies; a boat's
  hull identification number counts as a serial number (trackable).

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

# ----------------------------------------------------- test3 set (13, blind)
# Authored 2026-11-04 after corpus round 2 and candidate 2, without running the
# detector on any of it; the first run is recorded in training/README.md. Every
# name, street, city and organization is fresh: none appears in the generator
# pools or in the other gold sets, and the skill, language and vehicle words
# avoid the round-2 decoy pools, so the set measures the pattern, not the word.
# It mixes the layouts round 2 targeted (skill and language lists, year ranges,
# grades, a weekday before a quoted author, a vehicle model, chat greetings,
# prose and first-name-only names, varied birth labels) with layouts no training
# document has: a payslip whose employer is the letterhead, a UK plate and IBAN,
# an ISO birth date, facility names without the usual keyword (Academy,
# Institute, British "Centre"), a flat-before-street address, and plural
# pronouns that refer to people in one sentence and to parcels in the next.
TEST3 = [
 dict(id="gold_v3_01_cv_profile.txt",
  text=("CURRICULUM VITAE\nTheodora Lindahl\n"
        "Contact: t.lindahl@brevlada.example, +44 7700 900412\n"
        "Address: 9 Lantern Hill, Flat 2, Whitby, YO21 3AB\n"
        "Education: Ballymore Institute of Technology, B.Eng. Civil Engineering, "
        "2016.\n"
        "Experience: Site Engineer at Caldwell & Finch Surveyors (2016-2021); she "
        "coordinated three bridge inspections a year. Project Lead at Dunmore Rail "
        "Services since 2021.\n"
        "Languages: Welsh, Finnish, Greek.\nSkills: Revit, Primavera P6, Bluebeam."),
  redact=[("Theodora Lindahl","PERSON"),("t.lindahl@brevlada.example","EMAIL"),
          ("+44 7700 900412","PHONE"),("9 Lantern Hill","ADDRESS"),
          ("Flat 2","ADDRESS"),("Whitby","CITY_RESIDENCE"),("YO21 3AB","ZIP"),
          ("Ballymore Institute of Technology","SENSITIVE_ORG"),
          ("Caldwell & Finch Surveyors","EMPLOYER_ORG"),
          ("Dunmore Rail Services","EMPLOYER_ORG"),("she","PRONOUN")],
  keep=["Welsh","Finnish","Greek","Revit","Primavera P6","Bluebeam",
        "Site Engineer","Project Lead","2016","three bridge inspections"]),

 dict(id="gold_v3_02_quoted_email.txt",
  text=("From: imogen.hartley@corvusmedia.example\n"
        "To: callum.reyes@corvusmedia.example\n"
        "Subject: RE: launch rota\n\n"
        "Callum, the print run moves to Thursday. Can you confirm the Antwerp "
        "warehouse has the pallets? Rhiannon from dispatch will hold the courier "
        "slot.\n\n"
        "> On Wednesday Callum Reyes wrote:\n"
        "> Fine by me. His team covers the evening shift, so the van is loaded "
        "by 18:00.\n"),
  redact=[("imogen.hartley@corvusmedia.example","EMAIL"),
          ("callum.reyes@corvusmedia.example","EMAIL"),
          ("Callum Reyes","PERSON"),("Callum","PERSON"),("Rhiannon","PERSON"),
          ("his","PRONOUN")],
  keep=["Thursday","Wednesday","Antwerp","evening","18:00","launch rota",
        "print run","courier slot"]),

 dict(id="gold_v3_03_motor_renewal.txt",
  text=("MOTOR POLICY RENEWAL\nPolicyholder: Oluwaseun Adebayo\n"
        "Policy No.: POL 4471 9920 36\n"
        "Vehicle: Volvo V60, registration KX19 TRV, VIN YV1ZW74E0M1234567.\n"
        "Annual premium 612.40, renewal date 12/01/2026. No claims in the last "
        "five years.\n"
        "He can cancel within 14 days of renewal without charge. Courtesy car "
        "included."),
  redact=[("Oluwaseun Adebayo","PERSON"),("POL 4471 9920 36","POLICY_NUMBER"),
          ("KX19 TRV","LICENSE_PLATE"),("YV1ZW74E0M1234567","VIN"),
          ("he","PRONOUN")],
  keep=["Volvo V60","612.40","12/01/2026","five years","14 days",
        "Courtesy car"]),

 dict(id="gold_v3_04_live_chat.txt",
  text=("LIVE CHAT #90315\n"
        "[14:02] Support: Hello and welcome. How can we help today?\n"
        "[14:03] Baptiste Lacroix: Hi, order ORD-55-20931 arrived with a cracked "
        "lid.\n"
        "[14:04] Support: Sorry to hear that. Could you give me the customer "
        "number?\n"
        "[14:05] Baptiste Lacroix: Sure, CU-104772. The box left your Gothenburg "
        "hub on Saturday.\n"
        "[14:06] Support: Thanks. I have asked my colleague Anselm to ship a "
        "replacement tomorrow; he will email the new tracking link.\n"
        "[14:07] Baptiste Lacroix: Great, thanks."),
  redact=[("Baptiste Lacroix","PERSON"),("ORD-55-20931","ORDER_NUMBER"),
          ("CU-104772","GENERIC_ID"),("Anselm","PERSON"),("he","PRONOUN")],
  keep=["Hello","Hi","Sure","Great","Support","Gothenburg","Saturday","90315",
        "cracked lid","tracking link"]),

 dict(id="gold_v3_05_progress_report.txt",
  text=("PROGRESS REPORT\nFairhaven Community College\n"
        "Student: Wiktoria Zaremba, class 10-C\nStudent ID: FC-2026-0419\n"
        "Term average 78 percent; GPA 3.4 of 4.0. Attendance 96 percent.\n"
        "Wiktoria Zaremba transfers to Greystone Academy in September and her "
        "records follow by secure post. The form tutor commends her coursework "
        "in chemistry.\nIssued 07/03/2026."),
  redact=[("Wiktoria Zaremba","PERSON"),("10-C","GROUP_CLASS"),
          ("FC-2026-0419","GENERIC_ID"),("Greystone Academy","SENSITIVE_ORG"),
          ("her","PRONOUN")],
  keep=["Fairhaven Community College","78 percent","3.4","96 percent",
        "September","07/03/2026","form tutor","chemistry"]),

 dict(id="gold_v3_06_intake_record.txt",
  text=("INTAKE RECORD\nOakridge Counselling Service\n"
        "Client: Ferdinand Oyelaran (DOB 1983-07-19), client ref OC-2026-118.\n"
        "Referred by Marchmont Wellbeing Centre after an initial assessment.\n"
        "He prefers evening appointments and asked that letters go to his work "
        "address, 14 Corbett Street, Stirling.\n"
        "First session booked for 11/10/2026."),
  redact=[("Ferdinand Oyelaran","PERSON"),("1983-07-19","DOB"),
          ("OC-2026-118","GENERIC_ID"),
          ("Marchmont Wellbeing Centre","SENSITIVE_ORG"),("he","PRONOUN"),
          ("his","PRONOUN"),("14 Corbett Street","ADDRESS"),
          ("Stirling","CITY_RESIDENCE")],
  keep=["Oakridge Counselling Service","evening","11/10/2026",
        "initial assessment"]),

 dict(id="gold_v3_07_recommendation.txt",
  text=("LETTER OF RECOMMENDATION\nTo the selection committee,\n"
        "I supervised Rafael Castellano at Northgate Marine for four seasons, "
        "and I have rarely met a more careful deckhand. He completed the "
        "advanced rigging course in 2022 and later trained two apprentices. One "
        "of them, Nkechi, now leads her own crew.\n"
        "I recommend him without reservation. My direct line is 555-736-2208.\n"
        "Signed on 08/14/2026."),
  redact=[("Rafael Castellano","PERSON"),("Northgate Marine","EMPLOYER_ORG"),
          ("he","PRONOUN"),("them","PRONOUN"),("Nkechi","PERSON"),
          ("her","PRONOUN"),("him","PRONOUN"),("555-736-2208","PHONE")],
  keep=["deckhand","advanced rigging course","2022","four seasons",
        "08/14/2026","two apprentices","selection committee"]),

 dict(id="gold_v3_08_payslip.txt",
  text=("PAYSLIP \u2014 period 10/01/2026 to 10/31/2026\nPemberton Foods Ltd\n"
        "Employee: Siobhan Gallagher, employee no. EMP-30917\n"
        "Role: Line Supervisor, grade 4\n"
        "Gross pay 3,180.00; deductions 742.15; net pay 2,437.85.\n"
        "Paid by transfer to IBAN GB29 NWBK 6016 1331 9268 19 on 10/28/2026.\n"
        "Her holiday balance is 11 days. Queries go to the payroll desk."),
  redact=[("Siobhan Gallagher","PERSON"),("EMP-30917","GENERIC_ID"),
          ("GB29 NWBK 6016 1331 9268 19","IBAN"),("her","PRONOUN")],
  keep=["Pemberton Foods Ltd","Line Supervisor","grade 4","3,180.00",
        "2,437.85","10/28/2026","11 days","payroll desk"]),

 dict(id="gold_v3_09_volunteer_roster.txt",
  text=("VOLUNTEER ROSTER \u2014 harvest festival\n"
        "Saturday morning, gate and parking:\n"
        "- Anneliese Brink, 555-402-7719\n"
        "- Kwame Mensah, 555-402-8830\n"
        "Saturday afternoon, refreshments tent:\n"
        "- Ludmila Horakova, 555-402-9152\n"
        "Stewards wear the yellow vests from the storage container. The raffle "
        "draw follows the brass band in the main marquee; the prize list stays "
        "with the committee."),
  redact=[("Anneliese Brink","PERSON"),("555-402-7719","PHONE"),
          ("Kwame Mensah","PERSON"),("555-402-8830","PHONE"),
          ("Ludmila Horakova","PERSON"),("555-402-9152","PHONE")],
  keep=["Saturday","morning","afternoon","refreshments tent","yellow vests",
        "harvest festival","main marquee","brass band"]),

 dict(id="gold_v3_10_damage_claim.txt",
  text=("DAMAGE CLAIM\n"
        "Claimant: Gretchen Albrecht, 3 Tanner's Yard, Alnwick, NE66 1PQ\n"
        "Order number ORD-77-18204, tracking RR 445 221 879 GB, delivered "
        "09/26/2026.\n"
        "Item: espresso machine, serial number SN-7Q2M-88413, purchase price "
        "389.00.\n"
        "The carton arrived crushed on one side and the drip tray was cracked. "
        "She photographed the damage before unpacking and keeps the packaging.\n"
        "Resolution requested: replacement unit."),
  redact=[("Gretchen Albrecht","PERSON"),("3 Tanner's Yard","ADDRESS"),
          ("Alnwick","CITY_RESIDENCE"),("NE66 1PQ","ZIP"),
          ("ORD-77-18204","ORDER_NUMBER"),("SN-7Q2M-88413","SERIAL_NUMBER"),
          ("she","PRONOUN")],
  keep=["RR 445 221 879 GB","09/26/2026","389.00","espresso machine",
        "drip tray","replacement unit"]),

 dict(id="gold_v3_11_product_recall.txt",
  text=("PRODUCT RECALL NOTICE\n"
        "Halden Kitchenware recalls the 1.7-litre kettle, model HK-170, batch "
        "codes 2206 through 2211, sold between 03/01/2026 and 06/30/2026.\n"
        "The lid hinge can detach when the kettle is tilted. No injuries have "
        "been reported. Customers may return the kettle to any Bilbao or Antwerp "
        "branch for a refund of 34.99 or a replacement from the HK-180 range.\n"
        "Questions: the customer care team answers weekdays, 09:00 to 17:00."),
  redact=[],
  keep=["Halden Kitchenware","HK-170","2206","03/01/2026","06/30/2026",
        "Bilbao","Antwerp","34.99","HK-180","09:00","customer care team"]),

 dict(id="gold_v3_12_tenant_notice.txt",
  text=("NOTICE TO TENANTS\n"
        "Tenants: Elias Fenwick and Ruth Fenwick, Flat 6, 28 Harbour Lane, "
        "Dunmore.\n"
        "The boiler inspection takes place on 11/18/2026 between 09:00 and "
        "12:00. Their presence is not required; the engineer holds a key.\n"
        "Parcels left in the hallway will be moved to the bike store because "
        "they block the fire exit. The building's water supply is off for about "
        "an hour and its pressure may vary afterwards.\n"
        "Rent for December remains 1,150.00."),
  redact=[("Elias Fenwick","PERSON"),("Ruth Fenwick","PERSON"),
          ("Flat 6","ADDRESS"),("28 Harbour Lane","ADDRESS"),
          ("Dunmore","CITY_RESIDENCE"),("their","PRONOUN")],
  keep=["11/18/2026","09:00","12:00","they","its","bike store","fire exit",
        "1,150.00","December","engineer"]),

 dict(id="gold_v3_13_consular_appointment.txt",
  text=("CONSULAR APPOINTMENT CONFIRMATION\n"
        "Applicant: Mariana Castro, Brazilian passport FQ483921, born 23 August "
        "1979.\n"
        "Appointment: 12/09/2026 at the visa section, counter 4, for a short-stay "
        "visa; travelling to Porto on 01/15/2027 for a conference.\n"
        "Bring the printed form, one photo and proof of accommodation. She may "
        "bring an interpreter.\n"
        "Processing takes up to 15 working days."),
  redact=[("Mariana Castro","PERSON"),("FQ483921","PASSPORT"),
          ("23 August 1979","DOB"),("Porto","CITY_TRAVEL"),("she","PRONOUN")],
  keep=["Brazilian","12/09/2026","counter 4","01/15/2027","15 working days",
        "short-stay visa","conference"]),
]

# ----------------------------------------------------- test4 set (13, blind)
# Authored 2026-11-04 after the test3 fixes landed (commit 0de5ec7), without
# running the detector on any of it; the first run is recorded in
# training/README.md. Fresh names, streets, cities and organizations again. It
# probes how the test3 fixes generalize — facility words in new combinations
# (Technical Institute, Rehabilitation Centre) and new ones (Infirmary, Lyceum),
# other international phone shapes, a hyphenated policy prefix, a unit before
# the street, the new street types — and keeps measuring the known gaps: plural
# pronouns for objects near a person, an ID answered in chat without its label,
# a first name alone in prose, a CamelCase surname, a Dutch postcode, and a hub
# city inside a travel sentence.
TEST4 = [
 dict(id="gold_v4_01_staff_rota.txt",
  text=("WEEKLY STAFF ROTA\nPennine Freight Co.\n"
        "Monday: Amadou Diallo (07:00-15:00), Saoirse Brennan (15:00-23:00)\n"
        "Tuesday: Amadou Diallo (07:00-15:00), Leopold Krauss (15:00-23:00)\n"
        "Swap requests go to the duty manager by Friday noon. Overtime is paid "
        "at 1.5 times the hourly rate."),
  redact=[("Amadou Diallo","PERSON"),("Saoirse Brennan","PERSON"),
          ("Leopold Krauss","PERSON")],
  keep=["Pennine Freight Co.","07:00","15:00","23:00","Monday","Tuesday",
        "Friday","duty manager","1.5 times"]),

 dict(id="gold_v4_02_hospice_referral.txt",
  text=("REFERRAL\nRoyal Vale Infirmary\n"
        "Patient: Petronella Visser, date of birth 1958-11-23, hospital number "
        "RV-88-2201.\n"
        "Referred to St. Columba's Hospice for palliative care planning. She "
        "lives alone and her daughter visits daily.\n"
        "Transport arranged for 11/27/2026."),
  redact=[("Petronella Visser","PERSON"),("1958-11-23","DOB"),
          ("RV-88-2201","GENERIC_ID"),("St. Columba's Hospice","SENSITIVE_ORG"),
          ("she","PRONOUN"),("her","PRONOUN")],
  keep=["Royal Vale Infirmary","11/27/2026","palliative care planning",
        "daughter"]),

 dict(id="gold_v4_03_lyceum_transfer.txt",
  text=("TRANSFER CONFIRMATION\n"
        "Student: Ruaridh MacLeod, group 7-A, enrolled at Dunwich Lyceum from "
        "January.\n"
        "His previous school forwarded the attendance record and a reading-age "
        "assessment of 11.2 years.\n"
        "Uniform orders close on 12/15/2026; the office answers questions on "
        "weekday afternoons."),
  redact=[("Ruaridh MacLeod","PERSON"),("7-A","GROUP_CLASS"),
          ("Dunwich Lyceum","SENSITIVE_ORG"),("his","PRONOUN")],
  keep=["January","11.2 years","12/15/2026","afternoons","attendance record"]),

 dict(id="gold_v4_04_quoted_email_date.txt",
  text=("From: ottoline.marsh@solwaytimber.example\n"
        "To: kasimir.oyelowo@solwaytimber.example\n"
        "Subject: RE: kiln schedule\n\n"
        "Kasimir, the Genoa depot confirmed the Thursday slot. Thandiwe will "
        "send the loading list tonight; call her on +33 6 12 34 56 78 if it is "
        "late.\n\n"
        "> On Fri, 3 Oct 2026, Kasimir Oyelowo wrote:\n"
        "> Agreed. The kiln is down for maintenance until the 10th, so nothing "
        "ships before then.\n"),
  redact=[("ottoline.marsh@solwaytimber.example","EMAIL"),
          ("kasimir.oyelowo@solwaytimber.example","EMAIL"),
          ("Kasimir Oyelowo","PERSON"),("Kasimir","PERSON"),
          ("Thandiwe","PERSON"),("her","PRONOUN"),
          ("+33 6 12 34 56 78","PHONE")],
  keep=["Genoa","Thursday","kiln schedule","3 Oct 2026","10th","maintenance"]),

 dict(id="gold_v4_05_motor_quote.txt",
  text=("MOTOR INSURANCE QUOTE\nProposer: Henrik Solberg\n"
        "Vehicle: Nissan Qashqai, registration AB12 CDE, kept overnight on a "
        "driveway.\n"
        "Policy number AB-20394871 (previous insurer). Quoted premium 488.20 "
        "per year, excess 250.00.\n"
        "He has held a licence for 12 years with no convictions."),
  redact=[("Henrik Solberg","PERSON"),("AB12 CDE","LICENSE_PLATE"),
          ("AB-20394871","POLICY_NUMBER"),("he","PRONOUN")],
  keep=["Nissan Qashqai","488.20","250.00","12 years","driveway"]),

 dict(id="gold_v4_06_bank_notice.txt",
  text=("ACCOUNT NOTICE\nAccount holder: Dilnoza Karimova\n"
        "Correspondence address: Unit 4, 77 Quayside Walk, Kirkwall, KW15 1AB\n"
        "IBAN DE89 3704 0044 0532 0130 00\n"
        "From 01/01/2027 paper statements stop; they move to the online portal "
        "and their layout stays the same. She can opt out by replying within "
        "30 days."),
  redact=[("Dilnoza Karimova","PERSON"),("Unit 4","ADDRESS"),
          ("77 Quayside Walk","ADDRESS"),("Kirkwall","CITY_RESIDENCE"),
          ("KW15 1AB","ZIP"),("DE89 3704 0044 0532 0130 00","IBAN"),
          ("she","PRONOUN")],
  keep=["01/01/2027","they","their","online portal","30 days"]),

 dict(id="gold_v4_07_tenancy_end.txt",
  text=("END OF TENANCY STATEMENT\n"
        "Tenants: Tobias Achterberg and Zahra Mahmoudi\n"
        "Property: 12 Cobble Square, Flat 3, Tralee\n"
        "Deposit held 1,200.00; deductions 140.00 for carpet cleaning; refund "
        "1,060.00 within 10 working days.\n"
        "Keys were returned on 10/31/2026. Their forwarding address is on file."),
  redact=[("Tobias Achterberg","PERSON"),("Zahra Mahmoudi","PERSON"),
          ("12 Cobble Square","ADDRESS"),("Flat 3","ADDRESS"),
          ("Tralee","CITY_RESIDENCE"),("their","PRONOUN")],
  keep=["1,200.00","140.00","1,060.00","10 working days","10/31/2026",
        "carpet cleaning"]),

 dict(id="gold_v4_08_helpdesk_reply.txt",
  text=("HELPDESK CHAT #55120\n"
        "[11:40] Support: Hi there. What is the account number on your "
        "invoice?\n"
        "[11:41] Nnamdi Eze: It is 4471-8820-03.\n"
        "[11:42] Support: Thanks. The outage in the Malmo exchange ended at "
        "noon; your line should be back. My colleague Florentyna will text you "
        "once the test completes."),
  redact=[("Nnamdi Eze","PERSON"),("4471-8820-03","GENERIC_ID"),
          ("Florentyna","PERSON")],
  keep=["Hi there","Malmo","noon","55120","invoice","Support"]),

 dict(id="gold_v4_09_festival_debrief.txt",
  text=("FESTIVAL DEBRIEF\n"
        "Yesterday Casper Holm phoned about the marquee invoice and Marguerite "
        "Dubois dropped off the signed consent forms. Both agreed to help "
        "again next spring. She also suggested moving the raffle to the "
        "afternoon.\n"
        "Casper Holm left the hall at 7 o'clock in the evening after locking "
        "the storage container. The committee thanks everyone who stewarded "
        "the car park in the rain."),
  redact=[("Casper Holm","PERSON"),("Marguerite Dubois","PERSON"),
          ("she","PRONOUN"),("7 o'clock in the evening","TIME")],
  keep=["marquee invoice","next spring","afternoon","car park",
        "consent forms","storage container"]),

 dict(id="gold_v4_10_travel_authorisation.txt",
  text=("TRAVEL AUTHORISATION\n"
        "Traveller: Ingeborg Haugen, Norwegian passport No. P7731204, born on "
        "the 3rd of November 1988.\n"
        "Flying to Marseille on 02/03/2027 via the Lyon hub; return 02/10/2027. "
        "Hotel booking ref HB-99120 is attached.\n"
        "She has requested an aisle seat. Expenses are capped at 140.00 per "
        "day."),
  redact=[("Ingeborg Haugen","PERSON"),("P7731204","PASSPORT"),
          ("3rd of November 1988","DOB"),("Marseille","CITY_TRAVEL"),
          ("she","PRONOUN")],
  keep=["Norwegian","02/03/2027","Lyon","02/10/2027","HB-99120","140.00",
        "aisle seat"]),

 dict(id="gold_v4_11_rehab_progress.txt",
  text=("PROGRESS NOTE\n"
        "Client: Lucien Mercier (DoB 02-07-1985) completed the six-week "
        "programme at Brightwater Rehabilitation Centre and returns to work on "
        "12/01/2026.\n"
        "He attends a weekly check-in by phone, 555-610-4473, and keeps the "
        "exercise plan. Next review in March."),
  redact=[("Lucien Mercier","PERSON"),("02-07-1985","DOB"),
          ("Brightwater Rehabilitation Centre","SENSITIVE_ORG"),
          ("he","PRONOUN"),("555-610-4473","PHONE")],
  keep=["six-week programme","12/01/2026","March","exercise plan"]),

 dict(id="gold_v4_12_cv_institute.txt",
  text=("CV\nYevgenia Prokhorova\n"
        "41 Rosemary Parade, Haarlem, 2011 AB\n"
        "+31 20 234 5678 | y.prokhorova@postvak.example\n"
        "Education: Harbourside Technical Institute, mechanical engineering, "
        "from 2015 to 2019.\n"
        "Experience: Test Engineer at Aurelian Software since 2019; she "
        "maintains the rig fleet.\n"
        "Languages: Hebrew, Catalan, Maltese. Software: Simulink, SolidWorks, "
        "Ansys."),
  redact=[("Yevgenia Prokhorova","PERSON"),("41 Rosemary Parade","ADDRESS"),
          ("Haarlem","CITY_RESIDENCE"),("2011 AB","ZIP"),
          ("+31 20 234 5678","PHONE"),("y.prokhorova@postvak.example","EMAIL"),
          ("Harbourside Technical Institute","SENSITIVE_ORG"),
          ("Aurelian Software","EMPLOYER_ORG"),("she","PRONOUN")],
  keep=["Hebrew","Catalan","Maltese","Simulink","SolidWorks","Ansys",
        "Test Engineer","2015","2019","rig fleet"]),

 dict(id="gold_v4_13_harbour_notice.txt",
  text=("HARBOUR NOTICE\n"
        "Gale warning for the Esbjerg approaches from 18:00 tonight until "
        "06:00 tomorrow. The Genoa ferry is delayed by four hours and the "
        "Malmo crossing is cancelled.\n"
        "Berth fees for the night are waived. The harbour office reopens at "
        "08:00; call the duty line for berth allocation."),
  redact=[],
  keep=["Esbjerg","18:00","06:00","Genoa","Malmo","four hours","night",
        "08:00","duty line","harbour office"]),
]
# ------------------------------------------------- test5 set (13), blind batch
# Authored 2026-11-05 after the test4 fixes, without running the detector.
# Probes: names with lowercase particles, accents and inner capitals; thing
# labels other than vehicles; hub/branch cities inside travel sentences;
# Canadian and German postcodes; UK and German phone forms; staff/payroll
# numbers; facilities without a keyword (Nursery, Surgery, Language Centre);
# plural pronouns in both directions; the chat-reply ID again.
TEST5 = [
 dict(id="gold_v5_01_claim_form.txt",
  text=("INSURANCE CLAIM FORM\nClaimant: Fenna van der Berg\n"
        "Address: 220 Lakeshore Avenue, Toronto, ON M5V 2T6\n"
        "Policy No. HM 4471 2290 18; claim reference CL-77-0912.\n"
        "Staff No. 004512 at Maple Ridge Logistics (employer).\n"
        "Incident on 10/14/2026: a tree branch fell on the parked car. She "
        "reported it the same evening. The tyres were replaced in June; they "
        "are still under warranty."),
  redact=[("Fenna van der Berg","PERSON"),("220 Lakeshore Avenue","ADDRESS"),
          ("Toronto","CITY_RESIDENCE"),("M5V 2T6","ZIP"),
          ("HM 4471 2290 18","POLICY_NUMBER"),("CL-77-0912","GENERIC_ID"),
          ("004512","GENERIC_ID"),("Maple Ridge Logistics","EMPLOYER_ORG"),
          ("she","PRONOUN")],
  keep=["10/14/2026","tree branch","parked car","same evening","June","they",
        "under warranty"]),

 dict(id="gold_v5_02_nursery_letter.txt",
  text=("Sunrise Day Nursery\n"
        "Dear Ms Nowak,\n"
        "Zofia Nowak has settled in well since September. She enjoys the garden "
        "and the story corner.\n"
        "Her key worker is Beatriz Fernández, who can be reached on 020 7946 "
        "0958 during session hours.\n"
        "We have shared the allergy plan with Hillcrest Children's Clinic as "
        "agreed."),
  redact=[("Zofia Nowak","PERSON"),("Nowak","PERSON"),
          ("Beatriz Fernández","PERSON"),("020 7946 0958","PHONE"),
          ("Hillcrest Children's Clinic","SENSITIVE_ORG"),
          ("she","PRONOUN"),("her","PRONOUN")],
  keep=["Sunrise Day Nursery","September","garden","story corner",
        "session hours","allergy plan"]),

 dict(id="gold_v5_03_device_return.txt",
  text=("RETURN AUTHORISATION\n"
        "Customer: Siobhan O'Connell, customer number 3310-8847\n"
        "Device: Garmin Forerunner 255, serial number 7KX-220-4418\n"
        "Appliance: Miele Compact C1 (not covered by this return)\n"
        "Reason: the strap clip broke after two weeks. She asked for a "
        "replacement rather than a refund.\n"
        "Ship the device to Depot 7 in Leeds; the Leeds branch processes "
        "returns within five working days."),
  redact=[("Siobhan O'Connell","PERSON"),("3310-8847","GENERIC_ID"),
          ("7KX-220-4418","SERIAL_NUMBER"),("she","PRONOUN")],
  keep=["Garmin Forerunner 255","Miele Compact C1","Leeds","Depot 7",
        "two weeks","five working days","strap clip"]),

 dict(id="gold_v5_04_travel_itinerary.txt",
  text=("ITINERARY\n"
        "Passenger: Oluwaseun Adeyemi, birthday 14 Feb 1979\n"
        "Travels to Bergen on 03/02/2027 via the Oslo hub, returning from the "
        "Stavanger branch office on 03/06/2027.\n"
        "Contact during the trip: +41 44 668 18 00. He prefers a window seat.\n"
        "Flight SK 4123 is non-refundable; the Oslo lounge voucher is "
        "attached."),
  redact=[("Oluwaseun Adeyemi","PERSON"),("14 Feb 1979","DOB"),
          ("Bergen","CITY_TRAVEL"),("+41 44 668 18 00","PHONE"),
          ("he","PRONOUN")],
  keep=["Oslo","Stavanger","03/02/2027","03/06/2027","SK 4123","window seat",
        "lounge voucher"]),

 dict(id="gold_v5_05_reference_letter.txt",
  text=("To whom it may concern,\n"
        "I have known Adaeze Okonkwo for three years as her line manager at "
        "Fairhaven Analytics, where she joined as a junior analyst and now "
        "leads the reporting team.\n"
        "Adaeze handles difficult clients with patience. On one occasion she "
        "stayed late to rebuild a dashboard before a board meeting, and the "
        "directors noticed.\n"
        "I recommend her without reservation.\n"
        "Marcus Lindqvist\nHead of Analytics"),
  redact=[("Adaeze Okonkwo","PERSON"),("Adaeze","PERSON"),
          ("Fairhaven Analytics","EMPLOYER_ORG"),("her","PRONOUN"),
          ("she","PRONOUN"),("Marcus Lindqvist","PERSON")],
  keep=["Head of Analytics","three years","junior analyst","reporting team",
        "dashboard","board meeting","directors"]),

 dict(id="gold_v5_06_committee_minutes.txt",
  text=("MINUTES OF THE ALLOTMENT COMMITTEE\n"
        "Present: Hamid Rezaei, Greta Lindgren, Pieter de Vries.\n"
        "Hamid Rezaei and Greta Lindgren volunteered to repair the shared "
        "shed. They will buy the timber next week and submit their receipts "
        "to the treasurer.\n"
        "The water butts were inspected and found sound; the lids have been "
        "replaced.\n"
        "Next meeting on 11/19/2026 at the pavilion."),
  redact=[("Hamid Rezaei","PERSON"),("Greta Lindgren","PERSON"),
          ("Pieter de Vries","PERSON"),("they","PRONOUN"),("their","PRONOUN")],
  keep=["shared shed","timber","treasurer","water butts","11/19/2026",
        "pavilion"]),

 dict(id="gold_v5_07_gp_registration.txt",
  text=("NEW PATIENT REGISTRATION\n"
        "Name: Zsófia Németh, born 14.02.1979\n"
        "Home address: Hauptstrasse 12, 80331 Munich\n"
        "Previous practice: Lindenhof Surgery, transferred records on "
        "10/20/2026\n"
        "Current medication: none. She has no known allergies. Next of kin: "
        "Gábor Németh (brother)."),
  redact=[("Zsófia Németh","PERSON"),("14.02.1979","DOB"),
          ("Hauptstrasse 12","ADDRESS"),("80331","ZIP"),
          ("Munich","CITY_RESIDENCE"),("Lindenhof Surgery","SENSITIVE_ORG"),
          ("she","PRONOUN"),("Gábor Németh","PERSON")],
  keep=["10/20/2026","brother","medication","allergies"]),

 dict(id="gold_v5_08_chat_member.txt",
  text=("LIVE CHAT #90211\n"
        "[14:02] Agent: Welcome. Could you confirm your member number?\n"
        "[14:02] Priya Raghunathan: 88-1234-556, and my phone is 07700 "
        "900123.\n"
        "[14:03] Agent: Thank you. Your renewal is booked for 12/01/2026; the "
        "Glasgow branch will post the card.\n"
        "[14:04] Priya Raghunathan: Great, thanks."),
  redact=[("Priya Raghunathan","PERSON"),("88-1234-556","GENERIC_ID"),
          ("07700 900123","PHONE")],
  keep=["Glasgow","12/01/2026","renewal","Agent","90211","Welcome"]),

 dict(id="gold_v5_09_boat_sale.txt",
  text=("BILL OF SALE\nSeller: Declan McCarthy\nBuyer: Anneliese Baumgartner\n"
        "Vessel: Jeanneau Sun Odyssey 349, hull ID FR-JNO34912C626\n"
        "Engine: Yanmar 3YM30, 29 hp, serviced in May\n"
        "Price 98,500.00 paid by bank transfer to IBAN GB29 NWBK 6016 1331 "
        "9268 19.\n"
        "The sails were replaced in 2024; they carry a two-year warranty. He "
        "hands over the keys at the marina on 11/30/2026."),
  redact=[("Declan McCarthy","PERSON"),("Anneliese Baumgartner","PERSON"),
          ("FR-JNO34912C626","SERIAL_NUMBER"),
          ("GB29 NWBK 6016 1331 9268 19","IBAN"),("he","PRONOUN")],
  keep=["Jeanneau Sun Odyssey 349","Yanmar 3YM30","29 hp","May","98,500.00",
        "2024","they","two-year warranty","marina","11/30/2026"]),

 dict(id="gold_v5_10_payroll_memo.txt",
  text=("PAYROLL MEMO\n"
        "Employee: Bartholomew Oyelaran-Smith, payroll number P-204418, works "
        "three days a week at the Hull depot.\n"
        "Bank account 12345678, sort code 40-11-60. Salary paid on the 25th.\n"
        "His tax code changes to 1257L from April. Queries to "
        "payroll@northgate-foods.example."),
  redact=[("Bartholomew Oyelaran-Smith","PERSON"),("P-204418","GENERIC_ID"),
          ("12345678","BANK_ACCOUNT"),("40-11-60","BANK_ACCOUNT"),
          ("his","PRONOUN"),("payroll@northgate-foods.example","EMAIL")],
  keep=["Hull","three days a week","25th","1257L","April"]),

 dict(id="gold_v5_11_school_report.txt",
  text=("Greenbank Academy\nEND OF TERM REPORT\n"
        "Pupil: Tadhg Ó Briain, class 5-C\n"
        "Tadhg reads well above his age and has joined the chess club. He was "
        "late twice in October, arriving at 9 o'clock in the morning instead "
        "of 8:30.\n"
        "A referral to Riverside Speech and Language Centre was agreed with "
        "his parents."),
  redact=[("Tadhg Ó Briain","PERSON"),("Tadhg","PERSON"),("5-C","GROUP_CLASS"),
          ("his","PRONOUN"),("he","PRONOUN"),
          ("9 o'clock in the morning","TIME"),
          ("Riverside Speech and Language Centre","SENSITIVE_ORG")],
  keep=["Greenbank Academy","chess club","October","8:30","parents"]),

 dict(id="gold_v5_12_product_spec.txt",
  text=("DISHWASHER COMPARISON\n"
        "The Bosch Serie 4 works best on the eco cycle and finishes in 3 hours "
        "15 minutes. The Miele G 5000 is quieter at 44 dB but costs 180.00 "
        "more.\n"
        "Siemens offers a five-year parts warranty; AEG matches it only on the "
        "motor.\n"
        "All four models fit a standard 60 cm recess. Delivery to the Leeds "
        "and Bristol showrooms takes a week."),
  redact=[],
  keep=["Bosch Serie 4","Miele G 5000","Siemens","AEG","Leeds","Bristol",
        "3 hours 15 minutes","44 dB","180.00","60 cm","five-year"]),

 dict(id="gold_v5_13_quoted_thread.txt",
  text=("From: rhiannon.price@kestrelpress.example\n"
        "To: oisin.mcguinness@kestrelpress.example\n"
        "Subject: RE: proof corrections\n\n"
        "Oisín, the printer in Tallinn needs the corrected plates by "
        "Wednesday. Ring Hannelore on +49 (0)30 901820 if the courier slips; "
        "she is covering the desk this week.\n\n"
        "> On Tue, 27 Oct 2026, Oisín McGuinness wrote:\n"
        "> The map on page 12 still shows the old border. I will redraw it "
        "tonight.\n"),
  redact=[("rhiannon.price@kestrelpress.example","EMAIL"),
          ("oisin.mcguinness@kestrelpress.example","EMAIL"),
          ("Oisín McGuinness","PERSON"),("Oisín","PERSON"),
          ("Hannelore","PERSON"),("+49 (0)30 901820","PHONE"),
          ("she","PRONOUN")],
  keep=["Tallinn","Wednesday","27 Oct 2026","page 12","old border",
        "proof corrections","tonight","courier"]),
]


# ------------------------------------------------- test6 set (13), blind batch
# Authored 2026-11-07 after the test5 fixes (commit 60e4326) and the model
# move, without running the detector. Fresh names, streets, cities and
# organizations. Probes: more particles ('t, ap, Ní, bin/Al-) and Icelandic
# letters; ALL-CAPS form fields; initials and the Mx title; a chat transcript
# without brackets round the times; labels the rules may not know (NHS, PCN,
# DNI, driving licence, student number); dotted and ordinal birth dates; the
# 00 international prefix, Spanish and Norwegian phones; a German plate; a
# Spanish-order address and a Spanish school; facilities without a listed word
# (Care Home, Health Trust, Colegio); BIC and "card ending" as keeps; plural
# pronouns in both directions, including singular "they" for a named person;
# a travel city that also names an institute; and a no-PII site bulletin.
TEST6 = [
 dict(id="gold_v6_01_care_review.txt",
  text=("Hartswell Borough Council — Adult Social Care\n"
        "Annual review of care and support\n\n"
        "Resident: Mrs Beatrix Adebowale-Hart\n"
        "Date of birth: 03.11.1941\n"
        "NHS number: 943 476 5919\n"
        "Placement: Willowbrook Care Home, 4 Orchard Rise, Kingsmere KM2 7QL\n\n"
        "Mrs Adebowale-Hart was seen on 12 November 2026 with her daughter, Ms "
        "Funmi Adebowale. She is settled and uses the garden daily. The staff "
        "said they would review the mobility aids before winter. The community "
        "nurse from Kingsmere Health Trust visits weekly. Her contribution to "
        "the cost remains £214.60 a week under the Care Act 2014.\n\n"
        "Reviewing officer: Daniel Verhoeven, 01632 960 417, "
        "adultcare@hartswell.gov.uk"),
  redact=[("Beatrix Adebowale-Hart","PERSON"),("Adebowale-Hart","PERSON"),
          ("03.11.1941","DOB"),("943 476 5919","GENERIC_ID"),
          ("Willowbrook Care Home","SENSITIVE_ORG"),("4 Orchard Rise","ADDRESS"),
          ("Kingsmere","CITY_RESIDENCE"),("KM2 7QL","ZIP"),
          ("Funmi Adebowale","PERSON"),("she","PRONOUN"),("her","PRONOUN"),
          ("Kingsmere Health Trust","SENSITIVE_ORG"),
          ("Daniel Verhoeven","PERSON"),("01632 960 417","PHONE"),
          ("adultcare@hartswell.gov.uk","EMAIL")],
  keep=["Hartswell Borough Council","Adult Social Care","Annual review",
        "12 November 2026","garden","they","mobility aids","winter","weekly",
        "£214.60","Care Act 2014","Reviewing officer"]),

 dict(id="gold_v6_02_spanish_tenancy.txt",
  text=("Confirmación de reserva / Booking confirmation\n\n"
        "Tenant: María José García Lopez, DNI 48213976K\n"
        "Phone: +34 612 345 678\n"
        "Address of the property: Calle Mayor 12, 3ºB, 28013 Madrid\n"
        "Landlord: Sr. Joaquín de la Fuente\n"
        "Rent: €950 per month, deposit €1,900, from 1 December 2026 for twelve "
        "months.\n"
        "Rent is paid to IBAN ES91 2100 0418 4502 0005 1332 (BIC CAIXESBBXXX) "
        "on the first working day of each month.\n"
        "He will hand over two sets of keys at the property; they are labelled "
        "A and B. The building has a lift and a shared roof terrace."),
  redact=[("María José García Lopez","PERSON"),("48213976K","GENERIC_ID"),
          ("+34 612 345 678","PHONE"),("Calle Mayor 12, 3ºB","ADDRESS"),
          ("28013","ZIP"),("Madrid","CITY_RESIDENCE"),
          ("Joaquín de la Fuente","PERSON"),
          ("ES91 2100 0418 4502 0005 1332","IBAN"),("he","PRONOUN")],
  keep=["Booking confirmation","€950","€1,900","1 December 2026",
        "twelve months","CAIXESBBXXX","first working day","two sets of keys",
        "they","lift","shared roof terrace"]),

 dict(id="gold_v6_03_caps_application.txt",
  text=("APPLICATION FOR A SKILLED WORKER VISA — SECTION 3: APPLICANT DETAILS\n"
        "SURNAME: ONYEKWERE\n"
        "GIVEN NAMES: CHIDI EMEKA\n"
        "DATE OF BIRTH: 22/08/1987\n"
        "NATIONALITY: NIGERIAN\n"
        "PASSPORT NO: A08765432\n"
        "CURRENT ADDRESS: FLAT 9, 31 CASTLE STREET, DERBY DE1 3AB\n"
        "EMAIL: C.ONYEKWERE@EXAMPLE.COM\n"
        "TELEPHONE: +44 7700 900811\n"
        "EMPLOYER: TRENT VALLEY ENGINEERING LTD\n"
        "POSITION: WELDER\n"
        "APPLICATION REFERENCE: VAF-2026-118822\n"
        "I CONFIRM THAT THE INFORMATION GIVEN IS TRUE."),
  redact=[("ONYEKWERE","PERSON"),("CHIDI EMEKA","PERSON"),("22/08/1987","DOB"),
          ("A08765432","PASSPORT"),("FLAT 9, 31 CASTLE STREET","ADDRESS"),
          ("DERBY","CITY_RESIDENCE"),("DE1 3AB","ZIP"),
          ("C.ONYEKWERE@EXAMPLE.COM","EMAIL"),("+44 7700 900811","PHONE"),
          ("TRENT VALLEY ENGINEERING LTD","EMPLOYER_ORG"),
          ("VAF-2026-118822","GENERIC_ID")],
  keep=["SKILLED WORKER VISA","SECTION 3","APPLICANT DETAILS","SURNAME",
        "GIVEN NAMES","DATE OF BIRTH","NATIONALITY","NIGERIAN","CURRENT ADDRESS",
        "WELDER","APPLICATION REFERENCE","I CONFIRM"]),

 dict(id="gold_v6_04_chat_unbracketed.txt",
  text=("Brightwave Support — live chat transcript\n"
        "14:02 Agent (Sunita): Hi, thanks for contacting Brightwave Support. "
        "Could I take your membership number?\n"
        "14:03 Rhys ap Gruffudd: It's 7719-2034-88.\n"
        "14:03 Agent (Sunita): Thanks Rhys. And the model?\n"
        "14:04 Rhys ap Gruffudd: Model: Dyson V11, bought in March. The battery "
        "light blinks and it cuts out after two minutes.\n"
        "14:05 Agent (Sunita): I've raised ticket TK-55021. A replacement "
        "battery ships to 8 Bryn Teg, Caernarfon LL55 1AB within five working "
        "days. Order no. 20261107-4481 for your records."),
  redact=[("Rhys ap Gruffudd","PERSON"),("Rhys","PERSON"),("Sunita","PERSON"),
          ("7719-2034-88","GENERIC_ID"),("8 Bryn Teg","ADDRESS"),
          ("Caernarfon","CITY_RESIDENCE"),("LL55 1AB","ZIP"),
          ("20261107-4481","ORDER_NUMBER")],
  keep=["Brightwave Support","live chat transcript","membership number",
        "Dyson V11","March","battery light","two minutes","TK-55021",
        "five working days","14:02","14:04","14:05","your records"]),

 dict(id="gold_v6_05_appointment_sms.txt",
  text=("Two messages from Merlin Health Messaging\n\n"
        "1) Hi Ifeoma, your appointment at St Aldric's Hospital Fracture Clinic "
        "is on Tue 17 Nov at 09:40. Please bring your NHS number (ending 5919) "
        "and a list of current medicines. Call 0300 123 4567 to rebook. Reply "
        "STOP to opt out.\n\n"
        "2) Reminder for Bartek: flu jab walk-in at Thornbury Surgery, Thu "
        "10:00 to 12:00, no booking needed. Wear a short-sleeved top."),
  redact=[("Ifeoma","PERSON"),
          ("St Aldric's Hospital Fracture Clinic","SENSITIVE_ORG"),
          ("0300 123 4567","PHONE"),("Bartek","PERSON"),
          ("Thornbury Surgery","SENSITIVE_ORG")],
  keep=["Merlin Health Messaging","Tue 17 Nov","09:40","NHS number","5919",
        "current medicines","Reply STOP","flu jab","walk-in","10:00","12:00",
        "no booking needed","short-sleeved top"]),

 dict(id="gold_v6_06_mortgage_summary.txt",
  text=("Oakmere Mortgages — application summary (draft)\n\n"
        "Applicant 1: Mr Jan-Willem 't Hooft, born 7 March 1979\n"
        "Applicant 2: Mx Taylor Reid-Okafor, DOB 1981-11-30\n"
        "Current address (both): 12 Nightingale Walk, Apt 4, Chelmsford CM1 "
        "2QX — owned for 6 years\n"
        "Employer, Applicant 1: Fenland Water Services, salary £48,200\n"
        "Employer, Applicant 2: self-employed illustrator, average £31,400\n"
        "Repayments from sort code 20-45-18, account 71093364\n"
        "Credit card ending 4421 to be closed before completion\n"
        "Property: three-bedroom semi, valuation £412,000, loan to value 78%\n"
        "Notes: both applicants attended the meeting; they asked for a "
        "five-year fix."),
  redact=[("Jan-Willem 't Hooft","PERSON"),("7 March 1979","DOB"),
          ("Taylor Reid-Okafor","PERSON"),("1981-11-30","DOB"),
          ("12 Nightingale Walk, Apt 4","ADDRESS"),
          ("Chelmsford","CITY_RESIDENCE"),("CM1 2QX","ZIP"),
          ("Fenland Water Services","EMPLOYER_ORG"),
          ("20-45-18","BANK_ACCOUNT"),("71093364","BANK_ACCOUNT"),
          ("they","PRONOUN")],
  keep=["Oakmere Mortgages","application summary","Applicant 1","Applicant 2",
        "6 years","£48,200","self-employed illustrator","£31,400","4421",
        "three-bedroom semi","£412,000","78%","five-year fix","completion"]),

 dict(id="gold_v6_07_parking_notice.txt",
  text=("Eastvale City Council — Parking Services\n"
        "PENALTY CHARGE NOTICE\n\n"
        "To the registered keeper: Dr A. Quaye, 55 Ridgeway Court, Westbrook "
        "WB3 8HD\n"
        "Vehicle: Volkswagen Golf, grey, registration LB21 XYZ\n"
        "PCN number: EV 44021873\n"
        "Contravention: parked in a loading bay, Station Road car park, "
        "03/11/2026 at 18:52\n"
        "Penalty: £70, reduced to £35 if paid within 14 days of service.\n"
        "You may challenge this notice online or in writing. The CCTV footage "
        "is held for 90 days."),
  redact=[("A. Quaye","PERSON"),("55 Ridgeway Court","ADDRESS"),
          ("Westbrook","CITY_RESIDENCE"),("WB3 8HD","ZIP"),
          ("LB21 XYZ","LICENSE_PLATE"),("EV 44021873","GENERIC_ID")],
  keep=["Eastvale City Council","Parking Services","PENALTY CHARGE NOTICE",
        "registered keeper","Volkswagen Golf","grey","PCN number","loading bay",
        "Station Road car park","03/11/2026","18:52","£70","£35","14 days",
        "CCTV footage","90 days"]),

 dict(id="gold_v6_08_volunteer_email.txt",
  text=("From: Anjali Venkataraman <a.venkataraman@hollowbrook-foodbank.example>\n"
        "To: volunteers@hollowbrook-foodbank.example\n"
        "Subject: Saturday rota\n\n"
        "Hi all,\n"
        "Thanks for Saturday. Ms Delphine Rousseau-Mbeki from Larkfield Family "
        "Support Centre will join us at 10:00 to collect the parcels for their "
        "families; please have them ready by 09:45. Temitope said they can "
        "cover the van run after all.\n\n"
        "On Wed, 4 Nov 2026 at 16:20, Temitope <temi.a@example.com> wrote:\n"
        "> Can someone cover the van run? My car is in the garage and it won't "
        "be back until Friday.\n\n"
        "Best,\nAnjali"),
  redact=[("Anjali Venkataraman","PERSON"),("Anjali","PERSON"),
          ("a.venkataraman@hollowbrook-foodbank.example","EMAIL"),
          ("volunteers@hollowbrook-foodbank.example","EMAIL"),
          ("Delphine Rousseau-Mbeki","PERSON"),
          ("Larkfield Family Support Centre","SENSITIVE_ORG"),
          ("Temitope","PERSON"),("temi.a@example.com","EMAIL"),
          ("they","PRONOUN")],
  keep=["Saturday rota","Hi all","10:00","parcels","their","them","09:45",
        "van run","My car","garage","Friday","Best"]),

 dict(id="gold_v6_09_site_bulletin.txt",
  text=("SITE SAFETY BULLETIN — Felixstowe terminal and Doncaster depot\n\n"
        "Forklift models in service: Jungheinrich EFG 216, Toyota 8FBE18 and "
        "Hyster J2.5XN. Daily checks are due before the 06:00 shift. The "
        "Doncaster depot moves to the new racking layout on 1 December; the "
        "Felixstowe terminal follows in January. Incidents in October: two "
        "near-misses, no injuries. Report hazards on the yellow cards at the "
        "gatehouse or to the shift leads. The Zeebrugge hub is closed for "
        "maintenance on 15 November."),
  redact=[],
  keep=["Felixstowe terminal","Doncaster depot","Jungheinrich EFG 216",
        "Toyota 8FBE18","Hyster J2.5XN","06:00 shift","1 December","January",
        "October","near-misses","yellow cards","gatehouse","shift leads",
        "Zeebrugge hub","15 November"]),

 dict(id="gold_v6_10_trip_consent.txt",
  text=("Ashcombe Primary School\n"
        "Consent form — Year 4 trip to Bramblewood Nature Reserve, Thursday 26 "
        "November\n\n"
        "Pupil: Sigrún Hallgrímsdóttir, Class 4M\n"
        "Parent/carer: Mr Þórður Hallgrímsson, 07700 900214, "
        "thordur.h@example.is\n"
        "Medical notes: Sigrún carries an inhaler and she can self-administer.\n"
        "Emergency contact if the parent cannot be reached: Ms Caoimhe Ní "
        "Bhriain (aunt), 07700 900388\n"
        "Previous school: Hafnarfjörður International School (records "
        "transferred)\n"
        "Cost: £8.50, payable via the school app by 20 November. Packed lunch "
        "required."),
  redact=[("Sigrún Hallgrímsdóttir","PERSON"),("Sigrún","PERSON"),
          ("4M","GROUP_CLASS"),("Þórður Hallgrímsson","PERSON"),
          ("07700 900214","PHONE"),("thordur.h@example.is","EMAIL"),
          ("she","PRONOUN"),("Caoimhe Ní Bhriain","PERSON"),
          ("07700 900388","PHONE"),
          ("Hafnarfjörður International School","SENSITIVE_ORG")],
  keep=["Ashcombe Primary School","Consent form","Bramblewood Nature Reserve",
        "Thursday 26 November","inhaler","self-administer","Emergency contact",
        "aunt","records transferred","£8.50","school app","20 November",
        "Packed lunch"]),

 dict(id="gold_v6_11_physio_discharge.txt",
  text=("Brookfield Physiotherapy — discharge summary\n\n"
        "Patient: Mr Khalid bin Tariq Al-Farsi, DOB 15 Sep 1990\n"
        "Referred by: Dr Ingrid Haugland, Fernhill Sports Medicine Clinic\n"
        "Diagnosis: grade 2 hamstring strain (right)\n"
        "Sessions: six, from 2 Oct to 5 Nov 2026\n\n"
        "He has returned to full training. His club, Westfield Harriers, has "
        "been informed with his consent. He should continue the Nordic curls "
        "three times a week; the home exercise sheet is attached.\n"
        "Copy to: Dr P. Nwosu, Elm Street Medical Practice"),
  redact=[("Khalid bin Tariq Al-Farsi","PERSON"),("15 Sep 1990","DOB"),
          ("Ingrid Haugland","PERSON"),
          ("Fernhill Sports Medicine Clinic","SENSITIVE_ORG"),
          ("he","PRONOUN"),("his","PRONOUN"),("P. Nwosu","PERSON"),
          ("Elm Street Medical Practice","SENSITIVE_ORG")],
  keep=["Brookfield Physiotherapy","discharge summary",
        "grade 2 hamstring strain","right","six","2 Oct to 5 Nov 2026",
        "full training","Westfield Harriers","Nordic curls",
        "three times a week","home exercise sheet"]),

 dict(id="gold_v6_12_hire_damage.txt",
  text=("Vehicle hire — damage report\n"
        "Agreement: 55-20931\n"
        "Hirer: Ms Anne-Sophie de la Fontaine, born on the 3rd of March 1990\n"
        "Driving licence: FONTA812034AS9XY\n"
        "Vehicle: Peugeot 2008, registration M-RK 4471\n"
        "Returned to: our Hamburg Airport branch, 6 Nov 2026 at 08:15\n"
        "Damage noted: scuffed rear bumper, 12 cm; charge €180\n"
        "She disputes the charge and says the mark was present at pick-up from "
        "our Lübeck branch. Her photos from the pick-up are attached. Contact: "
        "+49 171 2345678. Her employer, Nordlicht Reisen GmbH, holds the "
        "company account used for the booking."),
  redact=[("55-20931","GENERIC_ID"),("Anne-Sophie de la Fontaine","PERSON"),
          ("3rd of March 1990","DOB"),("FONTA812034AS9XY","GENERIC_ID"),
          ("M-RK 4471","LICENSE_PLATE"),("she","PRONOUN"),("her","PRONOUN"),
          ("+49 171 2345678","PHONE"),("Nordlicht Reisen GmbH","EMPLOYER_ORG")],
  keep=["damage report","Peugeot 2008","Hamburg Airport branch","6 Nov 2026",
        "08:15","scuffed rear bumper","12 cm","€180","Lübeck branch","photos",
        "company account"]),

 dict(id="gold_v6_13_scholarship_letter.txt",
  text=("Halvorsen Foundation\n\n"
        "Dear Ms Castellanos-Reyes,\n\n"
        "We are pleased to award Lucía Castellanos-Reyes the 2027 Halvorsen "
        "Scholarship (student number 2027-04419). The award covers tuition at "
        "Trondheim Technical Institute for two years. Funds are paid to IBAN "
        "NO93 8601 1117 947 on 15 January and 15 August. She studied previously "
        "at Colegio San Esteban, whose reference we found compelling. Travel to "
        "Trondheim in January is reimbursed at economy rates. Please sign and "
        "return the acceptance by 1 December.\n\n"
        "Yours sincerely,\n"
        "Tomasz Wieczorek, Programme Director, 0047 55 12 34 56"),
  redact=[("Lucía Castellanos-Reyes","PERSON"),("Castellanos-Reyes","PERSON"),
          ("2027-04419","GENERIC_ID"),
          ("Trondheim Technical Institute","SENSITIVE_ORG"),
          ("NO93 8601 1117 947","IBAN"),("she","PRONOUN"),
          ("Colegio San Esteban","SENSITIVE_ORG"),("Trondheim","CITY_TRAVEL"),
          ("Tomasz Wieczorek","PERSON"),("0047 55 12 34 56","PHONE")],
  keep=["Halvorsen Foundation","2027 Halvorsen Scholarship","student number",
        "tuition","two years","15 January","15 August","economy rates",
        "1 December","acceptance","Programme Director"]),
]


# ------------------------------------------------- test7 set (13), blind batch
# Authored 2026-11-08 after the test6 fixes (commit 6c3f889), without running
# the detector. Fresh names, streets, cities and organizations. Probes the
# shapes those fixes assumed: greeting addressees that are not names
# (Householder, Finance, Committee, Visa Officer) beside greetings the rule
# does not list ("Good morning X,", "Hello X and Y,"); titled names under
# role labels (Dr., Prof., Rev., Sra., Herr, Dott.ssa, Eng.ª) and a company
# under one ("Registered keeper: ... Ltd"); 00-prefixed numbers that are not
# phones (a Polish postcode, an SSCC, an invoice number, a batch); ALL-CAPS
# fields other than names, with "MIDDLE NAME: NONE"; Romance-order addresses
# without a floor (Spanish, Italian, Portuguese) and "Via Royal Mail 48";
# hyphenated codes that look like German plates (USB-C 3.2, AC-DC 12 V,
# DIN-A 4, DE-HE 3, bay B-C 4) beside two real ones; a date range after
# "hire agreement"; the first-name echo against a city of the same name, a
# stop-listed first name and a three-letter one. New ground: a bracketed
# date-and-time chat export, Vietnamese and Polish names, a two-letter
# surname, a house-name address, an Eircode, an Italian fiscal code, plate
# and chassis number, a spaced National Insurance number, a slash-separated
# claim number, "half past six in the evening", and plural pronouns in both
# directions.
TEST7 = [
 dict(id="gold_v7_01_community_app.txt",
  text=("Four messages from the Larchfield Community App\n\n"
        "1) Dear Householder, the water will be off on Tuesday 17 November "
        "from 09:00 to 13:00 while the main in Tannery Road is replaced.\n\n"
        "2) Hi Finance, please pay the attached invoice INV-20417 for the "
        "lift service by Friday. Thanks, Bronagh\n\n"
        "3) Good morning Katarzyna, the parents' evening for class 6B has "
        "moved to Thursday at 17:30 in the main hall. Please tell Leszek as "
        "well.\n\n"
        "4) Dear Committee, I cannot attend on Thursday. Zdeněk Novotný from "
        "the roof group will present the three quotes in my place; he has "
        "the paperwork. Regards, Margaux"),
  redact=[("Bronagh","PERSON"),("Katarzyna","PERSON"),("6B","GROUP_CLASS"),
          ("Leszek","PERSON"),("Zdeněk Novotný","PERSON"),("he","PRONOUN"),
          ("Margaux","PERSON")],
  keep=["Larchfield Community App","Householder","Tuesday 17 November","09:00",
        "13:00","Tannery Road","Finance","INV-20417","lift service","Friday",
        "Good morning","parents' evening","Thursday","17:30","main hall",
        "Committee","roof group","three quotes","paperwork","Regards"]),

 dict(id="gold_v7_02_collision_report.txt",
  text=("Road traffic collision — report form\n"
        "Claim no. RTC/2026/004471\n\n"
        "Driver: Dr. Ngozi Chukwuemeka-Bell, date of birth 09/04/1984\n"
        "Driving licence number: CHUKW854094N99AB\n"
        "Vehicle: Seat Ibiza, registration KR19 PLM\n"
        "Registered keeper: Lexford Leasing Ltd\n"
        "Passenger: Prof. Alasdair MacKinnon\n"
        "Witness 1: Rev. Tobenna Achike, 07700 900552\n"
        "Witness 2: Sra. Inmaculada Ortega Ruiz, Avenida de Andalucía 8, "
        "41005 Sevilla, +34 954 55 01 23\n"
        "Other party: Herr Jürgen Weißmüller, insured with Albatros "
        "Versicherung, policy no. AV-7730219\n"
        "Location: junction of Mill Lane and the A614, 6 November 2026 at "
        "17:35\n\n"
        "The driver said she braked for a deer. Both witnesses confirm it; "
        "they were walking on the verge. No injuries were reported."),
  redact=[("RTC/2026/004471","GENERIC_ID"),
          ("Ngozi Chukwuemeka-Bell","PERSON"),("09/04/1984","DOB"),
          ("CHUKW854094N99AB","GENERIC_ID"),("KR19 PLM","LICENSE_PLATE"),
          ("Alasdair MacKinnon","PERSON"),("Tobenna Achike","PERSON"),
          ("07700 900552","PHONE"),("Inmaculada Ortega Ruiz","PERSON"),
          ("Avenida de Andalucía 8","ADDRESS"),("41005","ZIP"),
          ("Sevilla","CITY_RESIDENCE"),("+34 954 55 01 23","PHONE"),
          ("Jürgen Weißmüller","PERSON"),("AV-7730219","POLICY_NUMBER"),
          ("she","PRONOUN"),("they","PRONOUN")],
  keep=["Road traffic collision","report form","Driver","Seat Ibiza",
        "Registered keeper","Lexford Leasing Ltd","Passenger","Witness 1",
        "Witness 2","Other party","Albatros Versicherung","Mill Lane","A614",
        "6 November 2026","17:35","deer","verge","No injuries"]),

 dict(id="gold_v7_03_polish_delivery.txt",
  text=("Bursztyn Logistyka Sp. z o.o. — delivery note DN-118204\n\n"
        "Consignee: Agnieszka Wróblewska\n"
        "Deliver to: ul. Marszałkowska 84/12, 00-590 Warszawa\n"
        "Contact: 0048 22 555 01 47 or a.wroblewska@example.pl\n"
        "SSCC: 0035 9012 3400 0001 7421\n"
        "Tracking: 00359012340000017421\n"
        "Invoice no. 0042 0117 3381\n"
        "Goods: 2 cartons, 14.6 kg, batch 0031-22-456\n"
        "Driver's note: left with the concierge as she asked. Signed for at "
        "11:20 on 4 November 2026."),
  redact=[("Agnieszka Wróblewska","PERSON"),
          ("Marszałkowska 84/12","ADDRESS"),("00-590","ZIP"),
          ("Warszawa","CITY_RESIDENCE"),("0048 22 555 01 47","PHONE"),
          ("a.wroblewska@example.pl","EMAIL"),("she","PRONOUN")],
  keep=["Bursztyn Logistyka","delivery note","DN-118204","Consignee",
        "0035 9012 3400 0001 7421","00359012340000017421","0042 0117 3381",
        "2 cartons","14.6 kg","0031-22-456","concierge","11:20",
        "4 November 2026"]),

 dict(id="gold_v7_04_caps_registration.txt",
  text=("PATIENT REGISTRATION FORM — PLEASE COMPLETE IN BLOCK CAPITALS\n"
        "TITLE: MS\n"
        "FULL NAME: OLUWAFUNMILAYO ADESANYA-PRICE\n"
        "MIDDLE NAME: NONE\n"
        "PREVIOUS SURNAME: ADESANYA\n"
        "DATE OF BIRTH: 30 JUNE 1996\n"
        "SEX: FEMALE\n"
        "MARITAL STATUS: SINGLE\n"
        "OCCUPATION: STRUCTURAL ENGINEER\n"
        "COUNTRY OF BIRTH: GHANA\n"
        "ADDRESS: 18 SALTMARSH CLOSE, GRIMSBY DN32 9HX\n"
        "MOBILE: 07700 900671\n"
        "NEXT OF KIN: KWABENA ADESANYA (BROTHER), 07700 900672\n"
        "PREVIOUS GP: DR M. FAIRWEATHER, PRIORY LANE SURGERY\n"
        "NHS NUMBER: 485 777 3456\n"
        "ALLERGIES: PENICILLIN\n"
        "ORGAN DONOR: YES"),
  redact=[("OLUWAFUNMILAYO ADESANYA-PRICE","PERSON"),("ADESANYA","PERSON"),
          ("30 JUNE 1996","DOB"),("18 SALTMARSH CLOSE","ADDRESS"),
          ("GRIMSBY","CITY_RESIDENCE"),("DN32 9HX","ZIP"),
          ("07700 900671","PHONE"),("KWABENA ADESANYA","PERSON"),
          ("07700 900672","PHONE"),("M. FAIRWEATHER","PERSON"),
          ("PRIORY LANE SURGERY","SENSITIVE_ORG"),
          ("485 777 3456","GENERIC_ID")],
  keep=["PATIENT REGISTRATION FORM","BLOCK CAPITALS","TITLE","FULL NAME",
        "MIDDLE NAME","NONE","PREVIOUS SURNAME","DATE OF BIRTH","SEX","FEMALE",
        "MARITAL STATUS","SINGLE","OCCUPATION","STRUCTURAL ENGINEER",
        "COUNTRY OF BIRTH","GHANA","NEXT OF KIN","BROTHER","PREVIOUS GP",
        "NHS NUMBER","ALLERGIES","PENICILLIN","ORGAN DONOR"]),

 dict(id="gold_v7_05_italian_invoice.txt",
  text=("Officina Bellandi S.r.l. — Fattura / Invoice 2026/0931\n\n"
        "Cliente / Customer: Dott.ssa Fiammetta Carbone\n"
        "Indirizzo / Address: Via Garibaldi 27, 20121 Milano\n"
        "Codice fiscale: CRBFMT85T49F205U\n"
        "Telefono: +39 02 5550 0142\n"
        "Veicolo / Vehicle: Fiat Panda 1.2\n"
        "Targa / Plate: FX 482 LD\n"
        "Telaio / VIN: ZFA31200003456789\n"
        "Work: timing belt and water pump, 3.5 hours labour; 84,300 km on "
        "the clock\n"
        "Courtesy car: hire agreement 31/10/2026 to 03/11/2026, no charge\n"
        "Total: €412.80 including IVA at 22%\n"
        "Payable by bank transfer to IBAN IT60 X054 2811 1010 0000 0123 456 "
        "within 10 days.\n"
        "She collected the car on 3 November. Grazie!"),
  redact=[("Fiammetta Carbone","PERSON"),("Via Garibaldi 27","ADDRESS"),
          ("20121","ZIP"),("Milano","CITY_RESIDENCE"),
          ("CRBFMT85T49F205U","GENERIC_ID"),("+39 02 5550 0142","PHONE"),
          ("FX 482 LD","LICENSE_PLATE"),("ZFA31200003456789","VIN"),
          ("IT60 X054 2811 1010 0000 0123 456","IBAN"),("she","PRONOUN")],
  keep=["Officina Bellandi","Invoice","2026/0931","Customer","Codice fiscale",
        "Fiat Panda 1.2","timing belt","water pump","3.5 hours","84,300 km",
        "Courtesy car","31/10/2026","03/11/2026","no charge","€412.80","IVA",
        "22%","10 days","3 November","Grazie"]),

 dict(id="gold_v7_06_fleet_rollout.txt",
  text=("Fleet telematics rollout — depot notes, week 45\n\n"
        "Units fitted this week: 14 at the Kassel depot and 9 at the Leipzig "
        "hub. Each unit has two USB-C 3.2 ports, an AC-DC 12 V converter and "
        "Wi-Fi 6; the mounting template prints on DIN-A 4. Firmware region "
        "DE-HE 3 is loaded for Kassel and DE-SN 2 for Leipzig.\n\n"
        "Two vehicles are outstanding: KS-FT 2209 (the driver, Herr Yilmaz, "
        "is on leave until 16 November) and L-EV 318E, which is in the body "
        "shop. Bay B-C 4 is reserved for the fitter's van. The unit with "
        "serial number TLM-88213-A was returned faulty.\n\n"
        "Questions to the rollout desk on extension 4410."),
  redact=[("KS-FT 2209","LICENSE_PLATE"),("Yilmaz","PERSON"),
          ("L-EV 318E","LICENSE_PLATE"),("TLM-88213-A","SERIAL_NUMBER")],
  keep=["Fleet telematics rollout","depot notes","week 45","Kassel depot",
        "Leipzig hub","USB-C 3.2","AC-DC 12 V","Wi-Fi 6","DIN-A 4","DE-HE 3",
        "DE-SN 2","16 November","body shop","B-C 4","fitter's van",
        "rollout desk","extension 4410"]),

 dict(id="gold_v7_07_group_chat.txt",
  text=("WhatsApp chat export — Allotment keys\n"
        "[07/11/2026, 18:02] Mamadou Sy: Has anyone got the spare key for the "
        "tool shed?\n"
        "[07/11/2026, 18:05] Eilidh Farquharson: I gave it to Roisin on "
        "Sunday. She said she'd drop it back tonight.\n"
        "[07/11/2026, 18:06] Mamadou Sy: Thanks Eilidh. My number's changed "
        "btw, it's 07700 900934 now.\n"
        "[07/11/2026, 18:11] Roisin: Sorry both! It's at mine, 3 Weaver's "
        "Cottages. I'll bring it at 8.\n"
        "[07/11/2026, 18:12] Eilidh Farquharson: No rush. Membership renewals "
        "are due Friday, mine is AL-2291 if the treasurer is reading."),
  redact=[("Mamadou Sy","PERSON"),("Eilidh Farquharson","PERSON"),
          ("Eilidh","PERSON"),("Roisin","PERSON"),("she","PRONOUN"),
          ("07700 900934","PHONE"),("3 Weaver's Cottages","ADDRESS"),
          ("AL-2291","GENERIC_ID")],
  keep=["WhatsApp chat export","Allotment keys","spare key","tool shed",
        "Sunday","tonight","07/11/2026","18:02","18:11","No rush",
        "Membership renewals","Friday","treasurer"]),

 dict(id="gold_v7_08_portuguese_cv.txt",
  text=("CURRICULUM VITAE\n\n"
        "Name: Thiago Monteiro dos Santos\n"
        "Morada / Address: Rua Ferreira Borges 45, 3000-180 Coimbra\n"
        "Telemóvel / Mobile: +351 912 345 678\n"
        "E-mail: thiago.msantos@example.pt\n"
        "Date of birth: 12/05/1991\n"
        "Nationality: Brazilian\n"
        "Driving licence: category B, held since 2010\n\n"
        "Experience\n"
        "2019–2026  Site engineer, Construtora Ribeira Lda\n"
        "2015–2019  Junior surveyor, Granja & Filhos\n\n"
        "Education\n"
        "2015  MSc Civil Engineering, Instituto Politécnico do Mondego\n"
        "2009  Colégio São Bento\n\n"
        "Languages: Portuguese (native), English (C1), Spanish (B2)\n"
        "Referee: Eng.ª Leonor Vasconcelos, Construtora Ribeira Lda, "
        "leonor.v@ribeira.example"),
  redact=[("Thiago Monteiro dos Santos","PERSON"),
          ("Rua Ferreira Borges 45","ADDRESS"),("3000-180","ZIP"),
          ("Coimbra","CITY_RESIDENCE"),("+351 912 345 678","PHONE"),
          ("thiago.msantos@example.pt","EMAIL"),("12/05/1991","DOB"),
          ("Construtora Ribeira Lda","EMPLOYER_ORG"),
          ("Granja & Filhos","EMPLOYER_ORG"),
          ("Instituto Politécnico do Mondego","SENSITIVE_ORG"),
          ("Colégio São Bento","SENSITIVE_ORG"),
          ("Leonor Vasconcelos","PERSON"),
          ("leonor.v@ribeira.example","EMAIL")],
  keep=["CURRICULUM VITAE","Brazilian","category B","held since 2010",
        "Experience","Site engineer","Junior surveyor","Education",
        "MSc Civil Engineering","Languages","Portuguese","native","English",
        "C1","Spanish","B2","Referee"]),

 dict(id="gold_v7_09_sample_run_email.txt",
  text=("From: Florence Abernethy <florence.abernethy@castellomoda.example>\n"
        "To: Ana Beltrán Soler <ana.beltran@castellomoda.example>\n"
        "Cc: warehouse@castellomoda.example\n"
        "Subject: Sample run for the spring line\n\n"
        "Hi Ana,\n\n"
        "The sample run is confirmed. The cutters want the patterns by "
        "Wednesday; they work from paper, so please print them at full scale. "
        "The samples go to the Florence warehouse on Friday and Faith "
        "Oyinlola will sign them in. Faith is in from 07:30.\n\n"
        "Could you ring the dye house on 055 123 4567 and ask about the "
        "indigo? Their last batch was uneven.\n\n"
        "Thanks,\n"
        "Florence Abernethy\n"
        "Production Manager"),
  redact=[("Florence Abernethy","PERSON"),
          ("florence.abernethy@castellomoda.example","EMAIL"),
          ("Ana Beltrán Soler","PERSON"),("Ana","PERSON"),
          ("ana.beltran@castellomoda.example","EMAIL"),
          ("warehouse@castellomoda.example","EMAIL"),
          ("Faith Oyinlola","PERSON"),("Faith","PERSON"),
          ("055 123 4567","PHONE")],
  keep=["Sample run","spring line","cutters","patterns","Wednesday","they",
        "them","full scale","Florence warehouse","Friday","07:30","dye house",
        "indigo","Their","last batch","Thanks","Production Manager"]),

 dict(id="gold_v7_10_pension_statement.txt",
  text=("Tideswell Pension Scheme\n"
        "Member statement — year to 5 April 2026\n\n"
        "Member: Mr Emrys Llewellyn-Hughes\n"
        "Member ID: TPS/0044/71920\n"
        "Date of birth (DD/MM/YYYY): 02/09/1959\n"
        "National Insurance number: JT 48 13 92 B\n"
        "Spouse: Mrs Gwenllian Llewellyn-Hughes, d.o.b. 11.01.1962\n"
        "Address on file: The Old Forge, 6 Chapel Row, Bakewell DE45 1EW\n"
        "Normal retirement date: 2 September 2026\n"
        "Annual pension at that date: £9,418.20; lump sum option £41,300\n\n"
        "His nominated beneficiary is his spouse. They last updated the "
        "nomination in 2019. He does not need to take any action unless his "
        "details have changed.\n\n"
        "Queries: pensions helpline 0800 496 0143, weekdays 9 to 5."),
  redact=[("Emrys Llewellyn-Hughes","PERSON"),("TPS/0044/71920","GENERIC_ID"),
          ("02/09/1959","DOB"),("JT 48 13 92 B","UK_NINO"),
          ("Gwenllian Llewellyn-Hughes","PERSON"),("11.01.1962","DOB"),
          ("The Old Forge","ADDRESS"),("6 Chapel Row","ADDRESS"),
          ("Bakewell","CITY_RESIDENCE"),("DE45 1EW","ZIP"),
          ("his","PRONOUN"),("he","PRONOUN"),("they","PRONOUN"),
          ("0800 496 0143","PHONE")],
  keep=["Tideswell Pension Scheme","Member statement","5 April 2026",
        "Member ID","DD/MM/YYYY","Spouse","Normal retirement date",
        "2 September 2026","£9,418.20","£41,300","nominated beneficiary",
        "2019","pensions helpline","weekdays","9 to 5"]),

 dict(id="gold_v7_11_cardiology_letter.txt",
  text=("Wansbeck Valley NHS Foundation Trust\n"
        "Cardiology Outpatients\n\n"
        "Dear Dr Hetherington,\n\n"
        "Re: Mr Cuthbert Armstrong-Lowe, DOB 23 Aug 1948, hospital number "
        "H4471902, NHS no. 401 023 2137\n\n"
        "I reviewed this gentleman on 2 November 2026. He was referred from "
        "Coquet Medical Group after two falls at home. His echocardiogram at "
        "Alnmouth Community Hospital showed mild aortic stenosis. He lives "
        "alone in sheltered housing and attends Hadston Day Centre on "
        "Tuesdays; his daughter, Mrs Philippa Storey, collects his "
        "prescriptions from Boots. He does not want to move to a nursing "
        "home.\n\n"
        "Plan: ECG in six months; continue bisoprolol 2.5 mg. I have not "
        "changed his other medicines.\n\n"
        "Yours sincerely,\n"
        "Dr Soraya Haddad-Whitlock, Consultant Cardiologist\n"
        "Secretary: 01632 960 733"),
  redact=[("Hetherington","PERSON"),("Cuthbert Armstrong-Lowe","PERSON"),
          ("23 Aug 1948","DOB"),("H4471902","GENERIC_ID"),
          ("401 023 2137","GENERIC_ID"),("he","PRONOUN"),("his","PRONOUN"),
          ("Coquet Medical Group","SENSITIVE_ORG"),
          ("Alnmouth Community Hospital","SENSITIVE_ORG"),
          ("Hadston Day Centre","SENSITIVE_ORG"),
          ("Philippa Storey","PERSON"),("Soraya Haddad-Whitlock","PERSON"),
          ("01632 960 733","PHONE")],
  keep=["Wansbeck Valley NHS Foundation Trust","Cardiology Outpatients",
        "this gentleman","2 November 2026","two falls","echocardiogram",
        "mild aortic stenosis","sheltered housing","Tuesdays","prescriptions",
        "Boots","nursing home","ECG","six months","bisoprolol 2.5 mg",
        "Yours sincerely","Consultant Cardiologist","Secretary"]),

 dict(id="gold_v7_12_order_confirmation.txt",
  text=("Fernleaf Garden Supplies — order confirmation\n\n"
        "Hello Tevita and Mele,\n\n"
        "Thank you for your order no. FG-2026-774102, placed on 6 November "
        "2026.\n\n"
        "Item 1 — Name: Gardena Classic 30 hose reel; Qty 1; €64.95\n"
        "Item 2 — Name: Felco 2 secateurs; Qty 2; €119.80\n"
        "Dispatch: Via Royal Mail 48, tracked; tracking number RM 4471 2209 "
        "5GB\n"
        "Deliver to: Mele Taufa, 9 Sráid an Mhuilinn, Dingle, Co. Kerry, V92 "
        "X7K3\n"
        "Billing name: Tevita Taufa\n"
        "Paid with Visa ending 3310. Your loyalty card 6338 4410 0027 earned "
        "184 points.\n\n"
        "If the parcel has not arrived within five working days, email "
        "help@fernleaf.example or call 1800 555 014."),
  redact=[("Tevita","PERSON"),("Mele","PERSON"),
          ("FG-2026-774102","ORDER_NUMBER"),("Mele Taufa","PERSON"),
          ("9 Sráid an Mhuilinn","ADDRESS"),("Dingle","CITY_RESIDENCE"),
          ("V92 X7K3","ZIP"),("Tevita Taufa","PERSON"),
          ("6338 4410 0027","GENERIC_ID"),("help@fernleaf.example","EMAIL"),
          ("1800 555 014","PHONE")],
  keep=["Fernleaf Garden Supplies","order confirmation","6 November 2026",
        "Gardena Classic 30","hose reel","Felco 2","secateurs","Qty 1",
        "€64.95","€119.80","Dispatch","Royal Mail 48","RM 4471 2209 5GB",
        "Billing name","Visa ending","3310","184 points",
        "five working days"]),

 dict(id="gold_v7_13_invitation_letter.txt",
  text=("Aldergrove Instruments Ltd\n"
        "12 November 2026\n\n"
        "Dear Visa Officer,\n\n"
        "We invite Ms Nguyễn Thị Hồng Nhung (passport no. C4471902, born 18 "
        "March 1989), a senior buyer at our supplier Sông Hồng Precision Co., "
        "to visit our Sheffield plant from 7 to 18 December 2026. She will "
        "fly to Birmingham on 6 December. Her colleague Mr Trần Văn Khoa "
        "travels with her; his passport number is C5520318.\n\n"
        "We will meet them at the airport at half past six in the evening and "
        "cover all local costs. Their return flight is booked for 19 "
        "December.\n\n"
        "For verification call our HR desk on 0044 114 496 0827 or 0114 496 "
        "0827 (ext. 215).\n\n"
        "Yours faithfully,\n"
        "Maureen O'Dowd-Farrell\n"
        "Head of Procurement"),
  redact=[("Nguyễn Thị Hồng Nhung","PERSON"),("C4471902","PASSPORT"),
          ("18 March 1989","DOB"),("Sông Hồng Precision Co","EMPLOYER_ORG"),
          ("she","PRONOUN"),("Birmingham","CITY_TRAVEL"),("her","PRONOUN"),
          ("Trần Văn Khoa","PERSON"),("his","PRONOUN"),
          ("C5520318","PASSPORT"),("them","PRONOUN"),
          ("half past six in the evening","TIME"),("their","PRONOUN"),
          ("0044 114 496 0827","PHONE"),("0114 496 0827","PHONE"),
          ("Maureen O'Dowd-Farrell","PERSON")],
  keep=["Aldergrove Instruments Ltd","12 November 2026","Visa Officer",
        "senior buyer","Sheffield plant","7 to 18 December 2026","6 December",
        "airport","local costs","19 December","HR desk","ext. 215",
        "Yours faithfully","Head of Procurement"]),
]

# --------------------------------------------- blind held-out set test8 (13)
# Authored 2026-11-09 without running the detector. It probes what the test7
# fix round assumed: greeting and sign-off words outside the lists, a
# one-word company sign-off, German plates in a bay list with no vehicle
# wording, phones under a thing label, identifiers that are dates,
# block-capital role labels with values that are not names, "Via" and
# "Plaza" streets with no postcode, a street name with a function word,
# surnames that are company words, a bare "Name:" with a rare name and with
# products, CV lines with no year range, and house names with no house word.
TEST8 = [
 dict(id="gold_v8_01_tenants_board.txt",
  text=("Brackenridge Lettings — messages pinned to the Thornaby Wharf "
        "tenants' board\n\n"
        "1) Hiya Sunniva, the plumber is booked for Monday at 10. Cheers, "
        "Wojtek\n\n"
        "2) Dear Neighbours, the bin store will be jet-washed on Saturday "
        "morning. Please move your bikes. Kind regards, Brackenridge\n\n"
        "3) Morning Dafydd — your parcel is behind the meter cupboard. Ta, "
        "Gethin\n\n"
        "4) Hello Leaseholders, the AGM papers are attached. Any questions to "
        "the managing agent. All the best, Ottilie Varga\n\n"
        "5) Hey Caretaker, the light on the third-floor landing is out again. "
        "Thanks, Maintenance Committee"),
  redact=[("Sunniva","PERSON"),("Wojtek","PERSON"),("Dafydd","PERSON"),
          ("Gethin","PERSON"),("Ottilie Varga","PERSON")],
  keep=["Brackenridge Lettings","tenants' board","Hiya","plumber","Monday",
        "Cheers","Neighbours","bin store","Saturday morning","Kind regards",
        "Brackenridge","Morning","meter cupboard","Ta","Leaseholders",
        "AGM papers","managing agent","All the best","Caretaker",
        "third-floor landing","Maintenance Committee"]),

 dict(id="gold_v8_02_parking_bays.txt",
  text=("Hausverwaltung Lindenhof — Stellplatzliste / parking bay list, "
        "November 2026\n\n"
        "Bay 03 — M-KT 5038 — Fr. Annelise Bank\n"
        "Bay 07 — HH-RW 815 — Hr. Levent Karadağ-Vogt\n"
        "Bay 11 — free\n"
        "Bay 12 — B-ZL 2290E — Dr. Ioana Munteanu\n\n"
        "Notes: The ramp is certified to DIN-EN 1090; vehicles up to 3.5 t "
        "may use it. The charger at bay 12 is a WB-AC 22 wall unit. Gate "
        "remotes are booked to project code HV-LH 2026.\n"
        "Queries: Frau Bank chairs the residents' council, Tel. 089 5550 "
        "0173."),
  redact=[("M-KT 5038","LICENSE_PLATE"),("Annelise Bank","PERSON"),
          ("Bank","PERSON"),("HH-RW 815","LICENSE_PLATE"),
          ("Levent Karadağ-Vogt","PERSON"),("B-ZL 2290E","LICENSE_PLATE"),
          ("Ioana Munteanu","PERSON"),("089 5550 0173","PHONE")],
  keep=["Hausverwaltung Lindenhof","Stellplatzliste","parking bay list",
        "November 2026","Bay 03","Bay 07","Bay 11","free","ramp",
        "DIN-EN 1090","3.5 t","charger","WB-AC 22","wall unit","Gate remotes",
        "project code","HV-LH 2026","residents' council"]),

 dict(id="gold_v8_03_driver_manifest.txt",
  text=("Castlegate Couriers — driver manifest, run 14\n\n"
        "Stop 1\n"
        "Consignee: Mrs Oluchi Nwachukwu-Tait\n"
        "Address: Braeside, 4 Station Road, Pitlochry PH16 5AB\n"
        "Parcel contact: 07700 900318\n"
        "Tracking no.: CG 7720 4418 03\n"
        "Delivery note: 2026-11-06\n"
        "Customer reference: 06/11/2026\n\n"
        "Stop 2\n"
        "Consignee: Halvard Brekke\n"
        "Address: 14 Stoke by Nayland Road, Sudbury CO10 2XA\n"
        "Shipment phone: 01632 960 284\n"
        "Consignment: CN-55102, 3 cartons, 21 kg\n"
        "Case no. 2026-10-31/A17 (damage claim, his previous delivery)\n\n"
        "Driver notes: Stop 1 asked for a call ten minutes ahead. At stop 2 "
        "leave with the neighbour if he is out."),
  redact=[("Oluchi Nwachukwu-Tait","PERSON"),("Braeside","ADDRESS"),
          ("4 Station Road","ADDRESS"),("Pitlochry","CITY_RESIDENCE"),
          ("PH16 5AB","ZIP"),("07700 900318","PHONE"),
          ("Halvard Brekke","PERSON"),
          ("14 Stoke by Nayland Road","ADDRESS"),
          ("Sudbury","CITY_RESIDENCE"),("CO10 2XA","ZIP"),
          ("01632 960 284","PHONE"),("2026-10-31/A17","GENERIC_ID"),
          ("his","PRONOUN"),("he","PRONOUN")],
  keep=["Castlegate Couriers","driver manifest","run 14","Stop 1","Stop 2",
        "Consignee","CG 7720 4418 03","Delivery note","2026-11-06",
        "Customer reference","06/11/2026","CN-55102","3 cartons","21 kg",
        "damage claim","ten minutes","neighbour"]),

 dict(id="gold_v8_04_caps_tenancy.txt",
  text=("TENANCY APPLICATION — COMPLETE IN CAPITALS\n"
        "APPLICANT: ZSOFIA BARANYAI-COLE\n"
        "JOINT APPLICANT: NOT APPLICABLE\n"
        "GUARANTOR: TO BE CONFIRMED\n"
        "DATE OF BIRTH: 14 FEBRUARY 1993\n"
        "CURRENT ADDRESS: DUNROAMIN, 22 PARK ROAD, KENDAL LA9 4RT\n"
        "CURRENT LANDLORD: HOUSING ASSOCIATION\n"
        "EMPLOYER: WESTMORLAND DAIRIES LTD\n"
        "EMERGENCY CONTACT: IMRE BARANYAI (FATHER) 07700 900745\n"
        "NEXT OF KIN: AS ABOVE\n"
        "REFEREE: MRS P. ECCLESTONE, PREVIOUS LANDLADY\n"
        "PETS: ONE CAT\n"
        "SMOKER: NO\n"
        "NATIONAL INSURANCE NUMBER: PW 31 70 46 C\n"
        "SIGNED: Z. BARANYAI-COLE   DATE: 02/11/2026"),
  redact=[("ZSOFIA BARANYAI-COLE","PERSON"),("14 FEBRUARY 1993","DOB"),
          ("DUNROAMIN","ADDRESS"),("22 PARK ROAD","ADDRESS"),
          ("KENDAL","CITY_RESIDENCE"),("LA9 4RT","ZIP"),
          ("WESTMORLAND DAIRIES LTD","EMPLOYER_ORG"),
          ("IMRE BARANYAI","PERSON"),("07700 900745","PHONE"),
          ("P. ECCLESTONE","PERSON"),("PW 31 70 46 C","UK_NINO"),
          ("Z. BARANYAI-COLE","PERSON")],
  keep=["TENANCY APPLICATION","CAPITALS","APPLICANT","NOT APPLICABLE",
        "GUARANTOR","TO BE CONFIRMED","DATE OF BIRTH","CURRENT LANDLORD",
        "HOUSING ASSOCIATION","EMERGENCY CONTACT","FATHER","NEXT OF KIN",
        "AS ABOVE","REFEREE","PREVIOUS LANDLADY","PETS","ONE CAT","SMOKER",
        "SIGNED","02/11/2026"]),

 dict(id="gold_v8_05_secondment_flat.txt",
  text=("Subject: Keys for the secondment flat\n\n"
        "Hi Rosalind,\n\n"
        "The flat is confirmed: Via Saragozza 118, interno 4. No postcode on "
        "the contract, sorry. Signora Ginevra Tedeschi, the owner, lives next "
        "door at Via Saragozza 120 and holds the keys. She speaks a little "
        "English. Her mobile is 333 555 0199.\n\n"
        "I fly to Bologna on the 14th and should reach the flat by quarter to "
        "nine in the evening.\n\n"
        "For the March posting the agency offers Plaza de Olavide 6, 2ºB. I "
        "would take it; it is five minutes from the office.\n\n"
        "Please send the deposit receipt via email, not via the portal, and "
        "copy the Plaza Hotel booking to accounts for my first two nights.\n\n"
        "Best,\n"
        "Lorcan"),
  redact=[("Rosalind","PERSON"),("Via Saragozza 118","ADDRESS"),
          ("Ginevra Tedeschi","PERSON"),("Via Saragozza 120","ADDRESS"),
          ("she","PRONOUN"),("her","PRONOUN"),("333 555 0199","PHONE"),
          ("Bologna","CITY_TRAVEL"),
          ("quarter to nine in the evening","TIME"),
          ("Plaza de Olavide 6","ADDRESS"),("Lorcan","PERSON")],
  keep=["Subject","secondment flat","postcode","contract","owner",
        "a little English","the 14th","March posting","agency",
        "five minutes","office","deposit receipt","via email",
        "via the portal","Plaza Hotel","accounts","two nights","Best"]),

 dict(id="gold_v8_06_plain_cv.txt",
  text=("Curriculum vitae\n\n"
        "Name: Nkemdirim Ezeudu\n"
        "Address: Flat 2, 9 Marlinbank Road, Paisley PA3 4DG\n"
        "Phone: 07700 900486\n"
        "Email: n.ezeudu@example.org\n"
        "Born: 3 October 1990\n\n"
        "Work history\n"
        "Present: Warehouse team leader at Corbel Freight\n"
        "Earlier — Dispatcher, Tannahill Haulage (night shift)\n"
        "Before that, three years as a picker for Strathmore Wholesale\n"
        "Summer jobs: fruit picking, bar work\n\n"
        "Training\n"
        "HNC Supply Chain Management, Carrickstone College\n"
        "Forklift licence (counterbalance), renewed 2025\n"
        "First aid at work certificate, 2024\n\n"
        "Referees\n"
        "Morag Sutherland-Baird, operations manager, Corbel Freight — "
        "morag.sb@corbelfreight.example\n"
        "Available on request: a second referee from Tannahill Haulage."),
  redact=[("Nkemdirim Ezeudu","PERSON"),
          ("Flat 2, 9 Marlinbank Road","ADDRESS"),
          ("Paisley","CITY_RESIDENCE"),("PA3 4DG","ZIP"),
          ("07700 900486","PHONE"),("n.ezeudu@example.org","EMAIL"),
          ("3 October 1990","DOB"),("Corbel Freight","EMPLOYER_ORG"),
          ("Tannahill Haulage","EMPLOYER_ORG"),
          ("Strathmore Wholesale","EMPLOYER_ORG"),
          ("Carrickstone College","SENSITIVE_ORG"),
          ("Morag Sutherland-Baird","PERSON"),
          ("morag.sb@corbelfreight.example","EMAIL")],
  keep=["Curriculum vitae","Work history","Present","Warehouse team leader",
        "Dispatcher","night shift","three years","picker","Summer jobs",
        "fruit picking","bar work","Training",
        "HNC Supply Chain Management","Forklift licence","counterbalance",
        "renewed 2025","First aid at work certificate","Referees",
        "operations manager","Available on request"]),

 dict(id="gold_v8_07_occupational_health.txt",
  text=("Brindlemoor Occupational Health Ltd\n"
        "Occupational health report — confidential\n\n"
        "Employee: Thabo Nkosana-Fields   Payroll number: 0061447\n"
        "Manager: Lucinda Ashdown\n"
        "Date of assessment: 5 November 2026\n\n"
        "Thabo attended with his union representative, Mr Fintan O'Riordan. "
        "He has been absent since 12 October after surgery on his left "
        "shoulder at Fernbrook Orthopaedic Hospital. The surgeons were "
        "pleased with the result; they expect full movement within twelve "
        "weeks. Thabo and Lucinda agreed a phased return, and their next "
        "review is on 1 December.\n\n"
        "His GP, Dr Amara Okpara at Meadway Health Centre, has issued a fit "
        "note to 30 November. He should avoid lifting above 10 kg and working "
        "at night until then.\n\n"
        "Report prepared by Siobhán Kettlewell, occupational health adviser. "
        "Enquiries: 01632 960 517."),
  redact=[("Thabo Nkosana-Fields","PERSON"),("Thabo","PERSON"),
          ("0061447","GENERIC_ID"),("Lucinda Ashdown","PERSON"),
          ("Lucinda","PERSON"),("his","PRONOUN"),("he","PRONOUN"),
          ("Fintan O'Riordan","PERSON"),
          ("Fernbrook Orthopaedic Hospital","SENSITIVE_ORG"),
          ("their","PRONOUN"),("Amara Okpara","PERSON"),
          ("Meadway Health Centre","SENSITIVE_ORG"),
          ("Siobhán Kettlewell","PERSON"),("01632 960 517","PHONE")],
  keep=["Brindlemoor Occupational Health Ltd","Occupational health report",
        "confidential","Employee","Payroll number","Manager",
        "Date of assessment","5 November 2026","union representative",
        "12 October","left shoulder","surgeons","they","twelve weeks",
        "phased return","1 December","fit note","30 November","10 kg",
        "night","occupational health adviser"]),

 dict(id="gold_v8_08_support_log.txt",
  text=("Quillfield Broadband — support log, 4 November 2026\n\n"
        "09:14 Agent (Marisol): Thanks for calling Quillfield. Can I take the "
        "account holder's name?\n"
        "09:14 Caller: It's Barnaby Oosterhuis. The account is in my "
        "wife's name though, Liesbeth Oosterhuis.\n"
        "09:15 Agent (Marisol): No problem. And the account number or the "
        "landline?\n"
        "09:15 Caller: QB-0093-5562. The landline is 01632 960 902 but "
        "that's the thing that's dead.\n"
        "09:17 Agent (Marisol): I can see the router, serial number "
        "ZX4-7710-22918, went offline on 03/11/2026. I'll book an engineer "
        "for Thursday between 8 and 12.\n"
        "09:18 Caller: Fine. She works from home so someone will be in. Text "
        "me on 07700 900263 when he's on the way.\n"
        "09:19 Agent (Marisol): Done. The visit is logged for 05/11/2026. "
        "Anything else?\n"
        "09:19 Caller: No, cheers."),
  redact=[("Marisol","PERSON"),("Barnaby Oosterhuis","PERSON"),
          ("Liesbeth Oosterhuis","PERSON"),("QB-0093-5562","GENERIC_ID"),
          ("01632 960 902","PHONE"),("ZX4-7710-22918","SERIAL_NUMBER"),
          ("she","PRONOUN"),("07700 900263","PHONE"),("he","PRONOUN")],
  keep=["Quillfield Broadband","support log","4 November 2026","09:14",
        "Agent","Caller","account holder","account number","landline",
        "router","03/11/2026","engineer","Thursday","between 8 and 12",
        "works from home","05/11/2026","09:19","cheers"]),

 dict(id="gold_v8_09_refund_letter.txt",
  text=("Sparrowhawk Energy\n"
        "Refund confirmation\n\n"
        "Mr Joost van den Heuvel\n"
        "14 Mill Street\n"
        "Ludlow\n"
        "SY8 1BB\n\n"
        "Dear Mr van den Heuvel,\n\n"
        "We have closed your gas account 8841 2207 16 and refunded the credit "
        "balance of £183.47 to the bank account ending 2816 (IBAN GB96 SPHK "
        "4005 3012 3428 16). The refund reference is 2026-11-03 and the money "
        "should reach you within five working days.\n\n"
        "Your final meter reading was 04417 on 28 October 2026 (meter serial "
        "number G4A 00417 2209). If you move again, quote your customer "
        "number 720 411 8393.\n\n"
        "Yours sincerely,\n"
        "Customer Accounts\n"
        "Sparrowhawk Energy"),
  redact=[("Joost van den Heuvel","PERSON"),("van den Heuvel","PERSON"),
          ("14 Mill Street","ADDRESS"),("Ludlow","CITY_RESIDENCE"),
          ("SY8 1BB","ZIP"),("8841 2207 16","GENERIC_ID"),
          ("GB96 SPHK 4005 3012 3428 16","IBAN"),
          ("G4A 00417 2209","SERIAL_NUMBER"),
          ("720 411 8393","GENERIC_ID")],
  keep=["Sparrowhawk Energy","Refund confirmation","gas account","£183.47",
        "ending 2816","refund reference","2026-11-03","five working days",
        "final meter reading","04417","28 October 2026","customer number",
        "Yours sincerely","Customer Accounts"]),

 dict(id="gold_v8_10_fixture_email.txt",
  text=("From: Coach Solveig <solveig.eikeland@harriers.example>\n"
        "Subject: Saturday's fixture\n\n"
        "Hello Parents and Carers,\n\n"
        "Squad for Saturday at Otterburn Park, meet 09:15:\n"
        "Ayaan Chaudhry, Freya Sandberg-Moore, Kofi Asamoah, Matilda "
        "Brzezińska and Oscar.\n\n"
        "Kofi's mum has offered lifts; ring her on 07700 900611 if you need "
        "one. Freya is carrying an ankle strain, so she will play one half "
        "only. The referees are provided by the league and they need the team "
        "sheet by Friday noon.\n\n"
        "Oscar's registration is still pending: his form has no date of "
        "birth. Could his parents check it and bring his passport on "
        "Saturday?\n\n"
        "Yours in sport,\n"
        "Solveig Eikeland\n"
        "Head coach, Otterburn Harriers Juniors"),
  redact=[("Solveig","PERSON"),("solveig.eikeland@harriers.example","EMAIL"),
          ("Ayaan Chaudhry","PERSON"),("Freya Sandberg-Moore","PERSON"),
          ("Freya","PERSON"),("Kofi Asamoah","PERSON"),("Kofi","PERSON"),
          ("Matilda Brzezińska","PERSON"),("Oscar","PERSON"),
          ("her","PRONOUN"),("07700 900611","PHONE"),("she","PRONOUN"),
          ("his","PRONOUN"),("Solveig Eikeland","PERSON")],
  keep=["Subject","fixture","Parents and Carers","Squad","Saturday",
        "Otterburn Park","09:15","lifts","ankle strain","one half",
        "referees","league","they","team sheet","Friday noon","registration",
        "pending","date of birth","passport","Yours in sport","Head coach",
        "Otterburn Harriers Juniors"]),

 dict(id="gold_v8_11_attestation.txt",
  text=("ATTESTATION D'HÉBERGEMENT / PROOF OF ACCOMMODATION\n\n"
        "Je soussignée / I, the undersigned, Mme Clémence Aubertin-Roche, née "
        "le 27/08/1975, demeurant au 12 rue des Lilas, 21000 Dijon, certifie "
        "héberger à mon domicile / certify that I host at my home:\n\n"
        "M. Idrissa Coulibaly, né le 4 mars 1998, passeport n° 18FV20417, "
        "depuis le 1er septembre 2026.\n\n"
        "Carte d'identité / ID card no.: 880769204417\n"
        "Tél. : 06 39 98 01 84\n\n"
        "Fait à Dijon, le 2 novembre 2026\n"
        "Signature : C. Aubertin-Roche"),
  redact=[("Clémence Aubertin-Roche","PERSON"),("27/08/1975","DOB"),
          ("12 rue des Lilas","ADDRESS"),("21000","ZIP"),
          ("Dijon","CITY_RESIDENCE"),("Idrissa Coulibaly","PERSON"),
          ("4 mars 1998","DOB"),("18FV20417","PASSPORT"),
          ("880769204417","GENERIC_ID"),("06 39 98 01 84","PHONE"),
          ("C. Aubertin-Roche","PERSON")],
  keep=["ATTESTATION D'HÉBERGEMENT","PROOF OF ACCOMMODATION",
        "Je soussignée","the undersigned","certify","my home","passeport",
        "1er septembre 2026","Carte d'identité","ID card",
        "2 novembre 2026","Signature"]),

 dict(id="gold_v8_12_visitor_log.txt",
  text=("Visitor log — Greyfriars Data Centre, gate B, 6 November 2026\n\n"
        "Time in | Name | Host | Vehicle reg | Badge\n"
        "07:52 | Winifred Holding | R. Nakamura | YD21 KXT | V-0417\n"
        "08:10 | Chinedu Anozie | R. Nakamura | on foot | V-0418\n"
        "08:31 | Sr. Baltasar Quintero | Halima Yusuf | 6205 KLM | V-0419\n"
        "11:05 | Lotte | Halima Yusuf | 9-XKR-27 | V-0420\n\n"
        "Notes: Mr Anozie left his phone at the gate; he collected it at 12:40. "
        "The chiller engineers swapped two units in hall 3 and their van "
        "stayed in the loading bay until 16:00. Badge V-0420 was not "
        "returned."),
  redact=[("Winifred Holding","PERSON"),("R. Nakamura","PERSON"),
          ("YD21 KXT","LICENSE_PLATE"),("Chinedu Anozie","PERSON"),
          ("Anozie","PERSON"),("Baltasar Quintero","PERSON"),
          ("Halima Yusuf","PERSON"),("6205 KLM","LICENSE_PLATE"),
          ("Lotte","PERSON"),("9-XKR-27","LICENSE_PLATE"),
          ("his","PRONOUN"),("he","PRONOUN")],
  keep=["Visitor log","Greyfriars Data Centre","gate B","6 November 2026",
        "Time in","Host","Vehicle reg","Badge","07:52","on foot","12:40",
        "chiller engineers","two units","hall 3","their","loading bay",
        "16:00"]),

 dict(id="gold_v8_13_warranty_claim.txt",
  text=("Corvid Cycles — warranty claim\n\n"
        "Product\n"
        "Name: Corvid Tarn 29 hardtail\n"
        "Frame number: WTU 284 K 0931 M\n"
        "Purchased: 14 March 2025, order KC-250314-0082, from the Keswick "
        "store\n\n"
        "Owner\n"
        "Name: Xiomara Villanueva-Ortiz\n"
        "Email: xiomara.vo@example.net\n"
        "Daytime phone: (01632) 960044\n\n"
        "Fault\n"
        "Name of part: SR Suntour XCR 32 fork\n"
        "Description: the lockout lever snapped on the first ride after the "
        "service. The mechanic, Pim, says the fork was fine when he signed it "
        "off.\n\n"
        "Decision: replace under warranty. Approved by Store Manager, 7 "
        "November 2026. Returns label RL-77120 sent by email."),
  redact=[("WTU 284 K 0931 M","SERIAL_NUMBER"),
          ("KC-250314-0082","ORDER_NUMBER"),
          ("Xiomara Villanueva-Ortiz","PERSON"),
          ("xiomara.vo@example.net","EMAIL"),("(01632) 960044","PHONE"),
          ("Pim","PERSON"),("he","PRONOUN")],
  keep=["Corvid Cycles","warranty claim","Product","Corvid Tarn 29",
        "hardtail","Frame number","14 March 2025","Keswick store","Owner",
        "Daytime phone","Fault","SR Suntour XCR 32","fork","lockout lever",
        "first ride","mechanic","Store Manager","7 November 2026",
        "Returns label","RL-77120"]),
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
                + validate(TEST2, "test2") + validate(TEST3, "test3")
                + validate(TEST4, "test4") + validate(TEST5, "test5")
                + validate(TEST6, "test6") + validate(TEST7, "test7")
                + validate(TEST8, "test8"))
    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} validation errors — corpus not written")
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    emit(DEV, GOLD_DIR / "dev_docs.jsonl", GOLD_DIR / "dev_cases.jsonl")
    emit(TEST, GOLD_DIR / "test_docs.jsonl", GOLD_DIR / "test_cases.jsonl")
    emit(TEST2, GOLD_DIR / "test2_docs.jsonl", GOLD_DIR / "test2_cases.jsonl")
    emit(TEST3, GOLD_DIR / "test3_docs.jsonl", GOLD_DIR / "test3_cases.jsonl")
    emit(TEST4, GOLD_DIR / "test4_docs.jsonl", GOLD_DIR / "test4_cases.jsonl")
    emit(TEST5, GOLD_DIR / "test5_docs.jsonl", GOLD_DIR / "test5_cases.jsonl")
    emit(TEST6, GOLD_DIR / "test6_docs.jsonl", GOLD_DIR / "test6_cases.jsonl")
    emit(TEST7, GOLD_DIR / "test7_docs.jsonl", GOLD_DIR / "test7_cases.jsonl")
    emit(TEST8, GOLD_DIR / "test8_docs.jsonl", GOLD_DIR / "test8_cases.jsonl")
    everything = (DEV + TEST + TEST2 + TEST3 + TEST4 + TEST5 + TEST6 + TEST7
                  + TEST8)
    n_redact = sum(len(d["redact"]) for d in everything)
    n_keep = sum(len(d["keep"]) for d in everything)
    print(f"wrote {len(DEV)} dev + {len(TEST)} test + {len(TEST2)} test2 + "
          f"{len(TEST3)} test3 + {len(TEST4)} test4 + {len(TEST5)} test5 + "
          f"{len(TEST6)} test6 + {len(TEST7)} test7 + {len(TEST8)} test8 docs "
          f"({n_redact} redact spans, {n_keep} keep "
          f"spans) to {GOLD_DIR}")


if __name__ == "__main__":
    main()
