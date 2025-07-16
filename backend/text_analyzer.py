"""
Advanced text analysis for job posting authenticity detection
"""

import re
import nltk
from textblob import TextBlob
import logging
from collections import Counter
import string

logger = logging.getLogger(__name__)

class JobPostingTextAnalyzer:
    """Advanced text analyzer for job posting authenticity"""
    
    def __init__(self):
        self.suspicious_keywords = self._load_suspicious_keywords()
        self.positive_keywords = self._load_positive_keywords()
        self.company_indicators = self._load_company_indicators()
        
    def _load_suspicious_keywords(self):
        """Load keywords that indicate suspicious job postings"""
        return {
            'urgency': [
                'urgent', 'immediate', 'asap', 'right away', 'today only',
                'limited time', 'act fast', 'hurry', 'quick start'
            ],
            'unrealistic_promises': [
                'guaranteed', 'easy money', 'no experience needed',
                'work from home guaranteed', 'high salary guaranteed',
                'instant approval', 'automatic acceptance'
            ],
            'payment_requests': [
                'registration fee', 'processing fee', 'admin fee',
                'send money first', 'pay upfront', 'advance payment',
                'security deposit', 'training fee'
            ],
            'vague_descriptions': [
                'data entry', 'copy paste', 'simple work',
                'easy job', 'part time full time salary',
                'flexible timing', 'work when you want'
            ],
            'contact_red_flags': [
                'whatsapp only', 'telegram only', 'sms only',
                'no phone calls', 'email not required',
                'contact via social media only'
            ],
            'earning_claims': [
                'earn.*per day', 'daily income', 'weekly payment',
                'instant payment', 'same day payment',
                'earn.*from home', 'make.*money.*online'
            ]
        }
    
    def _load_positive_keywords(self):
        """Load keywords that indicate legitimate job postings"""
        return {
            'company_info': [
                'company name', 'established', 'founded', 'headquarters',
                'office address', 'company website', 'about us',
                'company profile', 'organization'
            ],
            'job_details': [
                'job description', 'responsibilities', 'duties',
                'role overview', 'position summary', 'key tasks',
                'job requirements', 'qualifications'
            ],
            'requirements': [
                'education', 'degree', 'experience required',
                'skills needed', 'qualifications', 'certification',
                'background check', 'references'
            ],
            'benefits': [
                'health insurance', 'medical benefits', 'retirement plan',
                'paid leave', 'vacation days', 'sick leave',
                'professional development', 'training provided'
            ],
            'process': [
                'interview process', 'application deadline',
                'selection process', 'hiring process',
                'background verification', 'document verification'
            ],
            'contact_info': [
                'hr department', 'human resources', 'hiring manager',
                'recruiter', 'contact person', 'email address',
                'phone number', 'office hours'
            ]
        }
    
    def _load_company_indicators(self):
        """Load indicators of legitimate companies"""
        return [
            'pvt ltd', 'private limited', 'corporation', 'corp',
            'inc', 'incorporated', 'llc', 'limited liability',
            'co.', 'company', 'enterprises', 'group',
            'technologies', 'solutions', 'services', 'systems'
        ]
    
    def analyze_text(self, text):
        """Comprehensive text analysis"""
        if not text or len(text.strip()) < 10:
            return self._create_low_confidence_result("Insufficient text content")
        
        # Clean and prepare text
        cleaned_text = self._clean_text(text)
        
        # Perform various analyses
        results = {
            'text_length': len(text),
            'word_count': len(cleaned_text.split()),
            'sentence_count': len(self._get_sentences(text)),
            'suspicious_analysis': self._analyze_suspicious_patterns(cleaned_text),
            'positive_analysis': self._analyze_positive_indicators(cleaned_text),
            'company_analysis': self._analyze_company_indicators(cleaned_text),
            'language_quality': self._analyze_language_quality(text),
            'structure_analysis': self._analyze_text_structure(text),
            'contact_analysis': self._analyze_contact_information(cleaned_text)
        }
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(results)
        
        # Generate final assessment
        assessment = self._generate_assessment(overall_score, results)
        
        return {
            'score': overall_score,
            'prediction': 'genuine' if overall_score > 0.5 else 'fake',
            'confidence': overall_score if overall_score > 0.5 else (1 - overall_score),
            'detailed_analysis': results,
            'assessment': assessment,
            'recommendations': self._generate_recommendations(results)
        }
    
    def _clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\@]', ' ', text)
        
        return text.strip()
    
    def _get_sentences(self, text):
        """Extract sentences from text"""
        try:
            blob = TextBlob(text)
            return [str(sentence) for sentence in blob.sentences]
        except:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _analyze_suspicious_patterns(self, text):
        """Analyze suspicious patterns in text"""
        found_patterns = {}
        total_score = 0
        
        for category, keywords in self.suspicious_keywords.items():
            found_in_category = []
            
            for keyword in keywords:
                if re.search(keyword, text):
                    found_in_category.append(keyword)
            
            if found_in_category:
                found_patterns[category] = found_in_category
                # Weight different categories differently
                if category == 'payment_requests':
                    total_score += len(found_in_category) * 0.3
                elif category == 'unrealistic_promises':
                    total_score += len(found_in_category) * 0.25
                else:
                    total_score += len(found_in_category) * 0.15
        
        return {
            'patterns_found': found_patterns,
            'total_suspicious_score': min(total_score, 1.0),
            'categories_affected': len(found_patterns)
        }
    
    def _analyze_positive_indicators(self, text):
        """Analyze positive indicators in text"""
        found_indicators = {}
        total_score = 0
        
        for category, keywords in self.positive_keywords.items():
            found_in_category = []
            
            for keyword in keywords:
                if re.search(keyword, text):
                    found_in_category.append(keyword)
            
            if found_in_category:
                found_indicators[category] = found_in_category
                # Weight different categories
                if category in ['company_info', 'job_details']:
                    total_score += len(found_in_category) * 0.2
                elif category in ['requirements', 'process']:
                    total_score += len(found_in_category) * 0.15
                else:
                    total_score += len(found_in_category) * 0.1
        
        return {
            'indicators_found': found_indicators,
            'total_positive_score': min(total_score, 1.0),
            'categories_present': len(found_indicators)
        }
    
    def _analyze_company_indicators(self, text):
        """Analyze company legitimacy indicators"""
        found_indicators = []
        
        for indicator in self.company_indicators:
            if re.search(indicator, text):
                found_indicators.append(indicator)
        
        # Check for specific company name patterns
        company_pattern = r'\b[A-Z][a-z]+ (?:pvt ltd|private limited|corporation|corp|inc|llc)\b'
        company_matches = re.findall(company_pattern, text, re.IGNORECASE)
        
        return {
            'company_type_indicators': found_indicators,
            'potential_company_names': company_matches,
            'has_company_structure': len(found_indicators) > 0,
            'company_score': min(len(found_indicators) * 0.2, 1.0)
        }
    
    def _analyze_language_quality(self, text):
        """Analyze language quality and grammar"""
        try:
            blob = TextBlob(text)
            
            # Basic metrics
            sentences = blob.sentences
            words = blob.words
            
            if len(sentences) == 0:
                return {'quality': 'poor', 'score': 0.2}
            
            avg_words_per_sentence = len(words) / len(sentences)
            
            # Check for very short or very long sentences
            sentence_lengths = [len(str(sentence).split()) for sentence in sentences]
            
            quality_score = 0.5  # Base score
            
            # Reasonable sentence length
            if 5 <= avg_words_per_sentence <= 20:
                quality_score += 0.2
            
            # Consistent sentence lengths
            if len(set(sentence_lengths)) > 1:  # Variety in sentence lengths
                quality_score += 0.1
            
            # Check for proper capitalization
            capitalized_sentences = sum(1 for s in sentences if str(s)[0].isupper())
            if capitalized_sentences / len(sentences) > 0.8:
                quality_score += 0.1
            
            # Determine quality level
            if quality_score >= 0.7:
                quality = 'good'
            elif quality_score >= 0.5:
                quality = 'fair'
            else:
                quality = 'poor'
            
            return {
                'quality': quality,
                'score': quality_score,
                'avg_words_per_sentence': avg_words_per_sentence,
                'sentence_count': len(sentences),
                'word_count': len(words)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing language quality: {e}")
            return {'quality': 'unknown', 'score': 0.3}
    
    def _analyze_text_structure(self, text):
        """Analyze text structure and organization"""
        lines = text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # Check for structured elements
        has_headers = any(line.isupper() or line.endswith(':') for line in non_empty_lines)
        has_bullets = any(line.strip().startswith(('•', '-', '*', '1.', '2.')) for line in non_empty_lines)
        has_sections = len(non_empty_lines) > 3
        
        structure_score = 0.3  # Base score
        
        if has_headers:
            structure_score += 0.2
        if has_bullets:
            structure_score += 0.2
        if has_sections:
            structure_score += 0.1
        
        return {
            'has_headers': has_headers,
            'has_bullet_points': has_bullets,
            'has_multiple_sections': has_sections,
            'line_count': len(non_empty_lines),
            'structure_score': min(structure_score, 1.0)
        }
    
    def _analyze_contact_information(self, text):
        """Analyze contact information quality"""
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        # Phone pattern (basic)
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
        phones = re.findall(phone_pattern, text)
        
        # Website pattern
        website_pattern = r'www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}|https?://[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
        websites = re.findall(website_pattern, text)
        
        # Address indicators
        address_indicators = ['address', 'street', 'city', 'state', 'zip', 'postal']
        has_address = any(indicator in text for indicator in address_indicators)
        
        contact_score = 0.2  # Base score
        
        if emails:
            contact_score += 0.3
        if phones:
            contact_score += 0.2
        if websites:
            contact_score += 0.2
        if has_address:
            contact_score += 0.1
        
        return {
            'emails_found': emails,
            'phones_found': phones,
            'websites_found': websites,
            'has_address_info': has_address,
            'contact_score': min(contact_score, 1.0)
        }
    
    def _calculate_overall_score(self, results):
        """Calculate overall authenticity score"""
        # Base score
        score = 0.5
        
        # Subtract for suspicious patterns
        score -= results['suspicious_analysis']['total_suspicious_score'] * 0.4
        
        # Add for positive indicators
        score += results['positive_analysis']['total_positive_score'] * 0.3
        
        # Add for company indicators
        score += results['company_analysis']['company_score'] * 0.1
        
        # Add for language quality
        score += (results['language_quality']['score'] - 0.5) * 0.1
        
        # Add for text structure
        score += (results['structure_analysis']['structure_score'] - 0.3) * 0.05
        
        # Add for contact information
        score += (results['contact_analysis']['contact_score'] - 0.2) * 0.05
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))
    
    def _generate_assessment(self, score, results):
        """Generate human-readable assessment"""
        if score > 0.8:
            level = "Very Likely Genuine"
        elif score > 0.6:
            level = "Likely Genuine"
        elif score > 0.4:
            level = "Uncertain"
        elif score > 0.2:
            level = "Likely Fake"
        else:
            level = "Very Likely Fake"
        
        return {
            'level': level,
            'score': score,
            'key_factors': self._identify_key_factors(results)
        }
    
    def _identify_key_factors(self, results):
        """Identify key factors affecting the assessment"""
        factors = []
        
        # Suspicious patterns
        if results['suspicious_analysis']['total_suspicious_score'] > 0.3:
            factors.append("Multiple suspicious patterns detected")
        
        # Positive indicators
        if results['positive_analysis']['total_positive_score'] > 0.5:
            factors.append("Strong positive indicators present")
        
        # Company information
        if results['company_analysis']['has_company_structure']:
            factors.append("Company structure information found")
        
        # Language quality
        if results['language_quality']['quality'] == 'poor':
            factors.append("Poor language quality")
        elif results['language_quality']['quality'] == 'good':
            factors.append("Good language quality")
        
        # Contact information
        if results['contact_analysis']['contact_score'] > 0.6:
            factors.append("Comprehensive contact information")
        elif results['contact_analysis']['contact_score'] < 0.3:
            factors.append("Limited contact information")
        
        return factors
    
    def _generate_recommendations(self, results):
        """Generate recommendations for users"""
        recommendations = []
        
        if results['suspicious_analysis']['total_suspicious_score'] > 0.3:
            recommendations.append("Be cautious: Multiple red flags detected")
        
        if not results['contact_analysis']['emails_found']:
            recommendations.append("Verify: No email contact provided")
        
        if results['company_analysis']['company_score'] < 0.2:
            recommendations.append("Research: Limited company information")
        
        if results['language_quality']['quality'] == 'poor':
            recommendations.append("Warning: Poor text quality may indicate fake posting")
        
        if not recommendations:
            recommendations.append("Appears legitimate, but always verify independently")
        
        return recommendations
    
    def _create_low_confidence_result(self, reason):
        """Create result for insufficient text"""
        return {
            'score': 0.2,
            'prediction': 'fake',
            'confidence': 0.8,
            'reason': reason,
            'assessment': {
                'level': 'Very Likely Fake',
                'score': 0.2,
                'key_factors': [reason]
            },
            'recommendations': ['Insufficient information to make reliable assessment']
        }
