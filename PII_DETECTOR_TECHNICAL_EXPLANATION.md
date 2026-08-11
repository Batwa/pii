# PII Detector - Technical Overview

## Overview
A privacy-supporting data-processing system that detects and redacts personally identifiable information (PII) from structured (CSV), unstructured (text documents), and visual (images) data formats. Output must be reviewed before it is shared or used for regulatory decisions.

## Technical Architecture

### 1. Multi-Engine Detection Framework

The system employs a **hybrid detection approach** combining multiple NLP and computer vision technologies:

#### For Text/Tabular Data:
- **Microsoft Presidio**: Primary detection engine using machine learning models to identify 20+ PII entity types (emails, phone numbers, SSNs, credit cards, names, addresses, etc.)
- **spaCy NLP**: Complementary Named Entity Recognition (NER) using the `en_core_web_lg` language model for contextual understanding
- **Custom Regex Patterns**: Pattern-based detection for domain-specific PII formats that may be missed by ML models
- **Ensemble Method**: Results from all three engines are deduplicated and merged using overlap detection algorithms with confidence scoring

#### For Image Data:
- **OpenCV Haar Cascades**: Multi-pass face detection with adaptive sensitivity for varying face sizes and orientations
- **Tesseract OCR**: Optical Character Recognition to extract text from images
- **Hybrid PII Analysis**: Extracted text is analyzed using both Presidio and regex patterns to identify PII within images

### 2. Smart Redaction Strategies

The system implements multiple redaction methodologies:

**For Text/Tabular Data:**
- **Masking**: Complete replacement with `****` or custom patterns
- **Pseudonymization**: Consistent fake data generation (same input → same pseudonym) using hash-based mapping
- **Partial Masking**: Preserves format while obscuring sensitive portions (e.g., "J*** S****" for names)
- **Label Replacement**: Semantic replacement with entity type labels (e.g., "[PERSON]")
- **Hash-based Anonymization**: Cryptographic hashing for one-way anonymization

**For Images:**
- **Gaussian Blur**: Multi-pass blurring with configurable intensity
- **Pixelation**: Block-based pixelation preserving image structure
- **Black Box Coverage**: Rectangular overlays with configurable padding

**For Structured Data:**
- **Column-Aware Processing**: Intelligent handling that preserves ID columns (customer_id, employee_id) while redacting sensitive data
- **Context-Aware Text Scanning**: Enhanced regex-based scanning for text-heavy columns (notes, comments, descriptions)

### 3. System Components

#### Core Detection Engine (`PIIDetector`)
- Pandas DataFrame integration for efficient batch processing
- Row-level and column-level PII scanning
- Configurable confidence thresholds
- Comprehensive error handling and logging

#### Advanced Text Processor (`TextPIIDetector`)
- Multi-file batch processing (TXT, JSON, MD, RTF, LOG)
- Real-time duplicate detection using spatial overlap algorithms
- Source attribution tracking (which engine detected which PII)
- Multiple simultaneous redaction strategy generation
- Detailed analytics and reporting

#### Targeted Redaction System (`TargetedPIIFixer`)
- Column metadata awareness (preserves structural identifiers)
- Selective field processing based on data types
- Enhanced regex patterns for embedded PII in text fields
- Before/after comparison reporting

#### Image Processing Pipeline (`ImagePIIDetector`)
- Multi-scale face detection with three detection passes
- Duplicate face removal using intersection-over-union (IoU) calculations
- OCR preprocessing with confidence filtering
- Coordinate-based region mapping for precise redaction
- Multi-format output generation (blur, pixelate, box)

### 4. Technical Implementation Details

**Technologies & Libraries:**
- **Python 3.13** - Core language
- **Pandas** - Data manipulation and DataFrame operations
- **Microsoft Presidio** - Enterprise-grade PII detection framework
- **spaCy** - Advanced NLP and NER models
- **OpenCV** - Computer vision and image processing
- **Tesseract OCR** - Text extraction from images
- **Uvicorn + HTML/CSS/JavaScript** - Local web interface and processing API
- **Regex Engine** - Custom pattern matching for domain-specific PII

**Key Algorithms:**
1. **Overlap Detection Algorithm**: Removes duplicate PII detections by calculating bounding box intersections and keeping highest-confidence results
2. **Pseudonym Mapping**: Uses MD5 hashing for consistent pseudonym generation (same entity → same pseudonym across files)
3. **Multi-Pass Face Detection**: Three detection passes with different sensitivity parameters to capture faces of varying sizes and orientations
4. **Spatial Deduplication**: Geometric overlap calculation for removing redundant detections from multiple engines

**Performance Optimizations:**
- Lazy initialization of NLP models
- Batch processing for multiple files
- Efficient DataFrame operations using vectorized pandas methods
- Configurable confidence thresholds to balance precision/recall

### 5. Privacy & Compliance Features

- **No Data Persistence**: Files processed in-memory and temporary storage only
- **Audit Trails**: Comprehensive logging of all detections and redactions
- **Multiple Redaction Options**: Flexibility to choose appropriate anonymization method per use case
- **Privacy workflow support**: Helps apply repeatable redaction steps, but does not itself establish regulatory compliance
- **Transparent Reporting**: Detailed reports showing what was detected, where, and how it was redacted

### 6. Software Engineering Practices

- **Modular Architecture**: Separate components for text, image, and structured data processing
- **Extensible Design**: Easy to add new PII types or redaction strategies
- **Error Handling**: Comprehensive exception handling with graceful degradation
- **Configuration Management**: Centralized configuration for thresholds, patterns, and file paths
- **Web Interface**: User-friendly local website for non-technical users

## Skills Demonstrated

- **Natural Language Processing**: Multi-engine NER, entity recognition, contextual understanding
- **Computer Vision**: Face detection, OCR, image preprocessing, region-based processing
- **Data Engineering**: Efficient DataFrame operations, batch processing, memory management
- **Machine Learning Integration**: Working with ML models (Presidio, spaCy) for production use
- **Software Architecture**: Modular design, separation of concerns, extensibility
- **Privacy Engineering**: Understanding of data anonymization, compliance requirements
- **Full-Stack Development**: Python backend with an HTML/CSS/JavaScript frontend
- **Algorithm Design**: Overlap detection, deduplication, pseudonymization algorithms

## Real-World Applications

- Research dataset preparation for public release
- Customer data anonymization for analytics
- Medical record de-identification
- Employee data sanitization for training/development
- Compliance automation for privacy regulations
