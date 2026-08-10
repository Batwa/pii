"""
Privacy Sandbox - PII Detection Engine
Tabular (CSV) data PII detection with smart and standard redaction modes
"""
import os
import pandas as pd
import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from config import (
    CONFIDENCE_THRESHOLD,
    REDACTION_METHODS,
    DATA_INPUT,
    DATA_OUTPUT,
    PII_COLUMN_KEYWORDS,
)


class RegexPIIMatch:
    """Lightweight stand-in for Presidio results from regex detection"""

    def __init__(self, entity_type):
        self.entity_type = entity_type


class PIIDetector:
    """Detect and redact PII in tabular datasets"""

    def __init__(self, confidence_threshold=None):
        """Initialize the PII detection engines"""
        print("🔍 Initializing Privacy Sandbox...")
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else CONFIDENCE_THRESHOLD
        )
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        self.preserve_columns = ['customer_id', 'employee_id', 'id', 'product_id']
        self.pii_column_keywords = PII_COLUMN_KEYWORDS
        self.additional_patterns = {
            'ssn_in_text': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'phone_in_text': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'email_in_text': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        }
        self.pattern_entity_types = {
            'ssn_in_text': 'US_SSN',
            'phone_in_text': 'PHONE_NUMBER',
            'email_in_text': 'EMAIL_ADDRESS',
        }

        print("✅ PII Detection engines ready!")

    def get_column_pattern(self, column_name):
        """Return the regex pattern for a column based on its name, if any"""
        column_lower = column_name.lower()
        for keyword, pattern_key in self.pii_column_keywords.items():
            if keyword in column_lower:
                return self.additional_patterns[pattern_key]
        return None

    def detect_pii_in_text(self, text, language='en'):
        """Detect PII in a single text string"""
        if not isinstance(text, str) or not text.strip():
            return []

        try:
            results = self.analyzer.analyze(
                text=text,
                language=language,
                score_threshold=self.confidence_threshold
            )
            return results
        except Exception as e:
            print(f"⚠️  Error analyzing text: {e}")
            return []

    def detect_regex_pii_in_text(self, text, pattern_key):
        """Detect PII in text using a named regex pattern"""
        if not isinstance(text, str) or not text.strip():
            return []

        pattern = self.additional_patterns.get(pattern_key)
        if pattern and pattern.search(text):
            return [RegexPIIMatch(self.pattern_entity_types[pattern_key])]
        return []

    def detect_pii_in_dataframe(self, df):
        """Detect PII in a pandas DataFrame"""
        pii_results = {}

        for column in df.columns:
            print(f"🔍 Scanning column: {column}")
            column_pii = []
            pattern_key = None
            column_lower = column.lower()
            for keyword, key in self.pii_column_keywords.items():
                if keyword in column_lower:
                    pattern_key = key
                    break

            for idx, value in df[column].astype(str).items():
                if pd.notna(value) and value != 'nan':
                    detected = self.detect_pii_in_text(value)

                    if pattern_key and not detected:
                        detected = self.detect_regex_pii_in_text(value, pattern_key)

                    if detected:
                        column_pii.append({
                            'row': idx,
                            'value': value,
                            'pii_detected': detected
                        })

            if column_pii:
                pii_results[column] = column_pii
                print(f"⚠️  Found {len(column_pii)} PII items in column '{column}'")
            else:
                print(f"✅ Column '{column}' is clean")

        return pii_results

    def partial_redact_value(self, value):
        """Partially redact a value, keeping first/last characters visible"""
        text = str(value)
        if len(text) > 4:
            return text[:2] + "***" + text[-1:]
        return REDACTION_METHODS['mask']

    def redact_dataframe(self, df, pii_results, method='mask'):
        """Redact PII in DataFrame based on detection results"""
        df_redacted = df.copy()

        for column, pii_items in pii_results.items():
            for item in pii_items:
                row = item['row']
                original_value = item['value']

                if method == 'mask':
                    df_redacted.loc[row, column] = REDACTION_METHODS['mask']
                elif method == 'partial':
                    df_redacted.loc[row, column] = self.partial_redact_value(original_value)
                elif method == 'replace':
                    df_redacted.loc[row, column] = REDACTION_METHODS['replace']
                else:
                    df_redacted.loc[row, column] = f"REDACTED_{hash(original_value) % 10000}"

        return df_redacted

    def smart_redact_text(self, text):
        """Apply regex-based redaction to free-text fields"""
        if not isinstance(text, str):
            return text

        redacted_text = text
        redacted_text = self.additional_patterns['ssn_in_text'].sub('[REDACTED-SSN]', redacted_text)
        redacted_text = self.additional_patterns['phone_in_text'].sub('[REDACTED-PHONE]', redacted_text)
        redacted_text = self.additional_patterns['email_in_text'].sub('[REDACTED-EMAIL]', redacted_text)
        return redacted_text

    def smart_detect_and_redact(self, df):
        """Smart redaction: preserve ID columns and scan free-text fields"""
        print("🧠 Smart PII Detection Starting...")

        pii_results = self.detect_pii_in_dataframe(df)
        df_redacted = df.copy()

        text_columns = ['notes', 'comments', 'description', 'remarks']
        pii_columns = list(self.pii_column_keywords.keys())

        for column, pii_items in pii_results.items():
            column_lower = column.lower()

            if any(preserve_col in column_lower for preserve_col in self.preserve_columns):
                print(f"🔒 Preserving ID column: {column}")
                continue

            if any(text_col in column_lower for text_col in text_columns):
                continue

            if any(keyword in column_lower for keyword in pii_columns):
                continue

            for item in pii_items:
                row = item['row']
                df_redacted.loc[row, column] = "****"

        for column in df.columns:
            column_lower = column.lower()

            if any(text_col in column_lower for text_col in text_columns):
                print(f"🔍 Applying enhanced text scanning to: {column}")
                df_redacted[column] = df_redacted[column].apply(self.smart_redact_text)
                continue

            if any(keyword in column_lower for keyword in pii_columns):
                pattern = self.get_column_pattern(column)
                if pattern is None:
                    continue

                print(f"🔍 Applying regex scanning to: {column}")
                for idx, value in df[column].astype(str).items():
                    if pd.notna(value) and value != 'nan' and pattern.search(value):
                        df_redacted.loc[idx, column] = "****"
                        column_pii = pii_results.setdefault(column, [])
                        if not any(item['row'] == idx for item in column_pii):
                            column_pii.append({
                                'row': idx,
                                'value': value,
                                'pii_detected': [RegexPIIMatch('REGEX_MATCH')]
                            })

        return df_redacted, pii_results

    def generate_smart_report(self, df_original, df_redacted, pii_results):
        """Generate a detailed comparison report"""
        print("\n📋 SMART REDACTION REPORT")
        print("=" * 50)

        total_changes = 0
        for column in df_original.columns:
            changes_in_column = 0
            for idx in range(len(df_original)):
                if str(df_original.iloc[idx][column]) != str(df_redacted.iloc[idx][column]):
                    changes_in_column += 1
                    total_changes += 1

            if changes_in_column > 0:
                print(f"📋 Column '{column}': {changes_in_column} values redacted")

        print("\n📊 Summary:")
        print(f"  Total values changed: {total_changes}")
        print(f"  Columns affected: {len(pii_results)}")
        print(f"  Rows processed: {len(df_original)}")


def main():
    """Test the PII detector with sample data"""
    print("🚀 Testing PII Detection...")

    sample_data = {
        'customer_id': ['C001', 'C002', 'C003'],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'email': ['john.doe@email.com', 'jane.smith@company.org', 'bob@example.net'],
        'phone': ['555-123-4567', '(555) 987-6543', '555.555.5555'],
        'notes': ['Called customer at john.doe@email.com', 'Meeting scheduled', 'Phone: 555-999-8888']
    }

    df = pd.DataFrame(sample_data)
    print("\n📊 Original Data:")
    print(df)

    detector = PIIDetector()

    print("\n--- Smart Redaction ---")
    df_smart, pii_results = detector.smart_detect_and_redact(df)
    print(df_smart)

    print("\n--- Complete Masking ---")
    pii_results = detector.detect_pii_in_dataframe(df)
    df_clean = detector.redact_dataframe(df, pii_results, method='mask')
    print(df_clean)


def test_with_csv():
    """Test smart redaction on customers.csv if available"""
    filepath = os.path.join(DATA_INPUT, "customers.csv")
    if not os.path.exists(filepath):
        print(f"No sample CSV at {filepath} — run with built-in sample data instead.")
        return

    detector = PIIDetector()
    df = pd.read_csv(filepath)
    df_smart, pii_results = detector.smart_detect_and_redact(df)
    detector.generate_smart_report(df, df_smart, pii_results)

    output_path = os.path.join(DATA_OUTPUT, "customers_smart_redacted.csv")
    os.makedirs(DATA_OUTPUT, exist_ok=True)
    df_smart.to_csv(output_path, index=False)
    print(f"\n💾 Saved: {output_path}")


if __name__ == "__main__":
    main()
