"""
Text File PII Detection - Detect and redact PII in text files, documents, JSON, etc.
Day 3: Text File PII Detection with advanced NER and pseudonymization
"""
import os
import json
import re
import hashlib
import random
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import spacy
from config import (
    CONFIDENCE_THRESHOLD, CUSTOM_NER_MODEL,
    PRESIDIO_SCORE_FLOOR,
)
from recognizers import build_analyzer_registry, passes_entity_threshold

class TextPIIDetector:
    """Advanced text file PII detection with multiple redaction strategies"""
    CUSTOM_ENTITY_TYPES = {
        "PERSON", "DATE_OF_BIRTH", "SENSITIVE_ORGANIZATION", "ADDRESS",
        "PHONE", "PHONE_RU", "SNILS", "INN", "BANK_ACCOUNT",
        "PAYMENT_CARD", "PASSPORT_NUMBER", "EMAIL", "EMAIL_ADDRESS", "BIC", "VIN",
        "LICENSE_PLATE", "POLICY_NUMBER", "STUDENT_ID",
    }

    # A pattern anchored to an explicit label outranks a bare digit-run match over
    # the same span: a labelled BIC beats a generic 9-digit SSN shape, and a
    # labelled INN beats Presidio's 12-digit Aadhaar recognizer.
    LABELLED_PATTERNS = {"bic", "inn_labelled"}
    PATTERN_ENTITY_TYPES = {"inn_labelled": "INN"}

    def __init__(self, confidence_threshold=CONFIDENCE_THRESHOLD):
        """Initialize text PII detector with multiple NLP engines"""
        print("📝 Initializing Text PII Detector...")
        
        self.confidence_threshold = confidence_threshold
        
        # Initialize Presidio (primary detection engine) with the country ID
        # recognizers Presidio ships but leaves disabled.
        self.analyzer = AnalyzerEngine(registry=build_analyzer_registry())
        self.anonymizer = AnonymizerEngine()
        
        # Initialize spaCy for advanced NER
        try:
            self.nlp = spacy.load("en_core_web_lg")
            self.spacy_available = True
            print("✅ spaCy NER model loaded")
        except OSError:
            print("⚠️  spaCy model not available, using Presidio only")
            self.spacy_available = False

        # A locally trained model is optional. It learns project-specific
        # entities such as sensitive healthcare organizations without sending
        # document data anywhere.
        try:
            self.custom_nlp = spacy.load(CUSTOM_NER_MODEL)
            self.custom_model_available = True
            print("✅ Custom PII model loaded")
        except OSError:
            self.custom_nlp = None
            self.custom_model_available = False
        
        # Supported file types
        self.supported_extensions = {'.txt', '.json', '.md', '.rtf', '.log'}
        
        # Pseudonymization mappings (consistent fake names for same person)
        self.pseudonym_cache = {}
        self.fake_names = [
            "Alex Johnson", "Jordan Smith", "Casey Brown", "Taylor Davis",
            "Morgan Wilson", "Riley Martinez", "Avery Garcia", "Quinn Anderson"
        ]
        
        # Enhanced regex patterns for text-specific PII
        self.text_patterns = {
            'email_address': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(?<!\w)(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})(?!\w)'),
            'phone_ru': re.compile(r'(?<!\w)(?:(?:\+7|8)[\s-]?(?:\(\d{3,4}\)|\d{3,4})[\s-]?\d{2,3}[\s-]\d{2}[\s-]\d{2})(?!\w)'),
            'ssn': re.compile(r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b'),
            'snils': re.compile(r'\b\d{3}-\d{3}-\d{3}\s\d{2}\b'),
            'inn': re.compile(r'\b\d{10}(?:\d{2})?\b'),
            'inn_labelled': re.compile(r'\b(?:INN|ИНН)\s*:?\s*(?P<pii>\d{10}(?:\d{2})?)\b', re.IGNORECASE),
            'bank_account': re.compile(r'\b[1-9]\d{19}\b'),
            'payment_card': re.compile(r'\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b'),
            'passport_number': re.compile(r'\b\d{2}\s\d{2}\sNo\.\s\d{6}\b', re.IGNORECASE),
            'bic': re.compile(r'\b(?:BIC|БИК)\s*:?[\s]*(?P<pii>\d{9})\b', re.IGNORECASE),
            'vin': re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b'),
            'license_plate': re.compile(r'\b[A-Z]\s?\d{3}\s?[A-Z]{2}\s?\d{2,3}\b'),
            'policy_number': re.compile(r'\b(?:policy|insurance\s+policy)\s+No\.?:?\s*(?P<pii>\d{2}(?:\s*\d{4}){3}\s*\d{2})\b', re.IGNORECASE),
            'student_id': re.compile(r'\bstudent\s+ID\s+No\.?:?\s*(?P<pii>[A-Z]{2,10}-\d{4}-\d{4,})\b', re.IGNORECASE),
            'credit_card': re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'date_of_birth': re.compile(r'\b(?:date\s+of\s+birth|born|dob)\s*[:,-]?\s*(?P<pii>(?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12][0-9]|3[01])[-/.](?:19|20)\d{2})\b', re.IGNORECASE),
            'sensitive_organization': re.compile(r'(?P<pii>(?:[A-Z][A-Za-z-]*\s+){0,5}(?:Hospital|Clinic|Polyclinic|Medical Center)(?:\s+No\.?\s*\d+)?)'),
            'address_line': re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b', re.IGNORECASE)
        }
        
        print("✅ Text PII Detector ready!")
    
    def read_text_file(self, file_path):
        """Read various text file formats"""
        _, ext = os.path.splitext(file_path.lower())
        
        if ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {ext}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            if ext == '.json':
                # For JSON, we'll process it as both structured data and text
                try:
                    json_data = json.loads(content)
                    return content, json_data
                except json.JSONDecodeError:
                    return content, None
            
            return content, None
            
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}")
    
    def detect_pii_presidio(self, text):
        """Detect PII using Microsoft Presidio"""
        try:
            # Query at a low floor, then apply the per-entity threshold. A single
            # global threshold would discard every pattern-only country recognizer.
            results = self.analyzer.analyze(
                text=text,
                language='en',
                score_threshold=PRESIDIO_SCORE_FLOOR
            )
            return [
                result for result in results
                if passes_entity_threshold(
                    result.entity_type, result.score, self.confidence_threshold
                )
            ]
        except Exception as e:
            print(f"⚠️  Presidio analysis failed: {e}")
            return []
    
    def detect_pii_spacy(self, text):
        """Detect PII using spaCy NER (complementary to Presidio)"""
        if not self.spacy_available:
            return []
        
        try:
            doc = self.nlp(text)
            spacy_entities = []
            
            for ent in doc.ents:
                # Map spaCy entity types to our PII categories
                pii_type = None
                if ent.label_ == "PERSON":
                    pii_type = "PERSON"
                elif ent.label_ in ["GPE", "LOC"]:  # Geopolitical entity, Location
                    pii_type = "LOCATION"
                elif ent.label_ == "ORG":
                    pii_type = "ORGANIZATION"
                elif ent.label_ == "DATE":
                    pii_type = "DATE_TIME"
                
                if pii_type:
                    spacy_entities.append({
                        'entity_type': pii_type,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'score': 0.9,  # spaCy doesn't provide confidence scores
                        'text': ent.text,
                        'source': 'spacy'
                    })
            
            return spacy_entities
            
        except Exception as e:
            print(f"⚠️  spaCy analysis failed: {e}")
            return []
    
    def detect_pii_regex(self, text):
        """Detect PII using regex patterns"""
        regex_results = []
        
        for pii_type, pattern in self.text_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                # Context patterns retain their surrounding text for matching,
                # but only the named PII span should be redacted.
                start, end = match.span('pii') if 'pii' in match.re.groupindex else match.span()
                if pii_type == 'sensitive_organization':
                    # Do not include a sentence verb such as “Visit” merely
                    # because it is capitalized at the beginning of a sentence.
                    leading = re.match(r'(?i)^(?:visit|at|the|from|to|for)\s+', text[start:end])
                    if leading:
                        start += leading.end()
                regex_results.append({
                    'entity_type': self.PATTERN_ENTITY_TYPES.get(pii_type, pii_type.upper()),
                    'start': start,
                    'end': end,
                    'score': 1.1 if pii_type in self.LABELLED_PATTERNS else 1.0,
                    'text': text[start:end],
                    'source': 'regex'
                })
        
        return regex_results

    def detect_pii_custom_model(self, text):
        """Detect project-specific PII categories using the local custom model."""
        if not self.custom_model_available:
            return []
        try:
            return [{
                'entity_type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'score': 0.95,
                'text': ent.text,
                'source': 'custom_ner',
            } for ent in self.custom_nlp(text).ents if ent.label_ in self.CUSTOM_ENTITY_TYPES]
        except Exception as e:
            print(f"⚠️  Custom PII model analysis failed: {e}")
            return []
    
    def comprehensive_pii_detection(self, text):
        """Combine all detection methods for comprehensive analysis"""
        all_results = []
        
        # 1. Presidio detection (primary)
        presidio_results = self.detect_pii_presidio(text)
        for result in presidio_results:
            all_results.append({
                'entity_type': result.entity_type,
                'start': result.start,
                'end': result.end,
                'score': result.score,
                'text': text[result.start:result.end],
                'source': 'presidio'
            })
        
        # 2. General spaCy detection (complementary)
        spacy_results = self.detect_pii_spacy(text)
        all_results.extend(spacy_results)

        # 3. Locally trained, project-specific entity model.
        all_results.extend(self.detect_pii_custom_model(text))

        # 4. Regex detection (catch specific patterns)
        regex_results = self.detect_pii_regex(text)
        all_results.extend(regex_results)

        # 5. Remove duplicates (same entity detected by multiple methods)
        unique_results = self.deduplicate_results(all_results, text)
        
        return unique_results
    
    def deduplicate_results(self, results, text):
        """Keep the strongest non-overlapping, single-line detections."""
        candidates = []
        for result in results:
            start, end = result.get('start'), result.get('end')
            if not isinstance(start, int) or not isinstance(end, int) or start >= end:
                continue
            span = text[start:end]
            # A span crossing a newline can merge document fields when masked.
            if '\n' in span or '\r' in span:
                continue
            # Generic dates such as “night” or a document issue date are not
            # redacted. Dates must have an explicit date-of-birth context.
            if result.get('entity_type') == 'DATE_TIME' and result.get('source') != 'custom_ner':
                continue
            if result.get('entity_type') == 'DATE_OF_BIRTH':
                context = text[max(0, start - 40):start]
                if not re.search(r'(?i)(?:born|date\s+of\s+birth|birth\s+date|dob)\s*[:,-]?\s*$', context):
                    continue
            candidates.append({**result, 'text': span})

        ranked = sorted(candidates, key=lambda item: (
            -item.get('score', 0),
            -(item['end'] - item['start']),
            item['start'],
        ))
        selected = []
        for result in ranked:
            if not any(result['start'] < item['end'] and result['end'] > item['start'] for item in selected):
                selected.append(result)
        return sorted(selected, key=lambda item: item['start'])
    
    def generate_pseudonym(self, original_text, entity_type):
        """Generate consistent pseudonyms for the same entity"""
        # Create a hash of the original text for consistency
        text_hash = hashlib.md5(original_text.encode()).hexdigest()[:8]
        
        if original_text in self.pseudonym_cache:
            return self.pseudonym_cache[original_text]
        
        if entity_type == "PERSON":
            pseudonym = f"Person_{text_hash}"
            if self.fake_names:
                pseudonym = random.choice(self.fake_names)
                self.fake_names.remove(pseudonym)  # Don't reuse names
        elif entity_type == "EMAIL_ADDRESS":
            pseudonym = f"user_{text_hash}@example.com"
        elif entity_type == "PHONE_NUMBER":
            pseudonym = f"555-{text_hash[:3]}-{text_hash[3:7]}"
        elif entity_type == "LOCATION":
            pseudonym = f"Location_{text_hash}"
        elif entity_type == "ORGANIZATION":
            pseudonym = f"Company_{text_hash}"
        else:
            pseudonym = f"{entity_type}_{text_hash}"
        
        self.pseudonym_cache[original_text] = pseudonym
        return pseudonym
    
    def apply_redaction_strategy(self, text, pii_results, strategy='mask'):
        """Apply different redaction strategies to the text"""
        if not pii_results:
            return text, {}
        
        # Sort by start position (descending) to avoid index shifting
        sorted_results = sorted(pii_results, key=lambda x: x['start'], reverse=True)
        
        redacted_text = text
        redaction_log = []
        
        for result in sorted_results:
            start, end = result['start'], result['end']
            original_text = result['text']
            entity_type = result['entity_type']
            
            if strategy == 'mask':
                replacement = "****"
            elif strategy == 'partial_mask':
                if len(original_text) > 4:
                    replacement = original_text[:2] + "***" + original_text[-1:]
                else:
                    replacement = "****"
            elif strategy == 'pseudonymize':
                replacement = self.generate_pseudonym(original_text, entity_type)
            elif strategy == 'label':
                replacement = f"[{entity_type}]"
            elif strategy == 'hash':
                replacement = f"HASH_{hashlib.md5(original_text.encode()).hexdigest()[:8]}"
            else:
                replacement = "****"
            
            # Apply replacement
            redacted_text = redacted_text[:start] + replacement + redacted_text[end:]
            
            # Log the redaction
            redaction_log.append({
                'original': original_text,
                'replacement': replacement,
                'entity_type': entity_type,
                'position': (start, end),
                'confidence': result['score'],
                'source': result.get('source', 'unknown')
            })
        
        return redacted_text, {'redactions': redaction_log}
    
    def process_text_file(self, file_path, redaction_strategies=['mask', 'pseudonymize', 'partial_mask']):
        """Process a single text file with multiple redaction strategies"""
        print(f"\n📝 Processing: {os.path.basename(file_path)}")
        
        # Read file
        try:
            content, json_data = self.read_text_file(file_path)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return None
        
        # Detect PII
        print("🔍 Detecting PII...")
        pii_results = self.comprehensive_pii_detection(content)
        
        # Analyze results
        analysis = {
            'filename': os.path.basename(file_path),
            'file_size': len(content),
            'total_pii_found': len(pii_results),
            'pii_by_type': {},
            'pii_by_source': {},
            'redacted_versions': {},
            'detection_details': pii_results
        }
        
        # Count PII by type and source
        for result in pii_results:
            pii_type = result['entity_type']
            source = result.get('source', 'unknown')
            
            analysis['pii_by_type'][pii_type] = analysis['pii_by_type'].get(pii_type, 0) + 1
            analysis['pii_by_source'][source] = analysis['pii_by_source'].get(source, 0) + 1
        
        print(f"📊 Found {len(pii_results)} PII entities")
        for pii_type, count in analysis['pii_by_type'].items():
            print(f"   - {pii_type}: {count}")
        
        # Apply different redaction strategies
        for strategy in redaction_strategies:
            print(f"🔒 Applying {strategy} redaction...")
            redacted_text, redaction_info = self.apply_redaction_strategy(content, pii_results, strategy)
            analysis['redacted_versions'][strategy] = {
                'content': redacted_text,
                'redaction_info': redaction_info
            }
        
        return analysis
    
    def save_redacted_files(self, analysis, output_dir):
        """Save all redacted versions and reports"""
        base_name = os.path.splitext(analysis['filename'])[0]
        saved_files = []
        
        # Save each redacted version
        for strategy, data in analysis['redacted_versions'].items():
            output_filename = f"{base_name}_{strategy}.txt"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(data['content'])
            
            saved_files.append(output_path)
            print(f"💾 Saved: {output_filename}")
        
        # Save detailed analysis report
        report_path = os.path.join(output_dir, f"{base_name}_analysis_report.json")
        
        # Prepare report (remove content to avoid huge JSON files)
        report_data = {
            **analysis,
            'redacted_versions': {
                strategy: data['redaction_info']
                for strategy, data in analysis['redacted_versions'].items()
            },
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        saved_files.append(report_path)
        print(f"📋 Analysis report saved: {os.path.basename(report_path)}")
        
        return saved_files
    
    def generate_summary_report(self, analysis):
        """Generate human-readable summary report"""
        report = f"""
TEXT FILE PII DETECTION REPORT
==============================
File: {analysis['filename']}
Size: {analysis['file_size']} characters

PII DETECTION SUMMARY:
Total PII entities found: {analysis['total_pii_found']}

PII BY TYPE:
"""
        
        for pii_type, count in analysis['pii_by_type'].items():
            report += f"  - {pii_type}: {count}\n"
        
        report += f"\nDETECTION SOURCES:\n"
        for source, count in analysis['pii_by_source'].items():
            report += f"  - {source}: {count}\n"
        
        report += f"\nREDACTION STRATEGIES APPLIED:\n"
        for strategy in analysis['redacted_versions'].keys():
            redactions = len(analysis['redacted_versions'][strategy]['redaction_info']['redactions'])
            report += f"  - {strategy}: {redactions} redactions applied\n"
        
        if analysis['detection_details']:
            report += f"\nDETAILED FINDINGS:\n"
            for i, detail in enumerate(analysis['detection_details'][:10], 1):  # Show first 10
                report += f"  {i}. {detail['entity_type']}: '{detail['text']}' (confidence: {detail['score']:.2f}, source: {detail.get('source', 'unknown')})\n"
            
            if len(analysis['detection_details']) > 10:
                report += f"  ... and {len(analysis['detection_details']) - 10} more\n"
        
        return report
