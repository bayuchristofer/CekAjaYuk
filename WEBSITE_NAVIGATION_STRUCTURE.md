# 🗺️ STRUKTUR NAVIGASI WEBSITE CEKAJAYUK

## 📋 **SITEMAP & NAVIGATION STRUCTURE**

### **🏠 MAIN NAVIGATION**
```
CekAjaYuk Website
├── 🏠 Beranda (#home)
├── ℹ️ Tentang (#about)
└── ⚙️ Cara Kerja (#how-it-works)
```

### **📱 PAGE SECTIONS**

#### **1. 🏠 HEADER & NAVIGATION**
- **Logo**: CekAjaYuk dengan shield icon
- **Navigation Menu**:
  - Beranda → Hero Section
  - Tentang → About Section (belum diimplementasi)
  - Cara Kerja → How It Works Section (belum diimplementasi)

#### **2. 🎯 HERO SECTION (#home)**
- **Judul**: "Deteksi Iklan Lowongan Kerja Palsu dengan AI"
- **Deskripsi**: Penjelasan singkat tentang layanan
- **System Status Dashboard**:
  - 🤖 Backend Status
  - 🧠 Models Status
  - 📊 Dataset Status
  - 🔍 OCR Status
  - 🔄 Refresh Button
- **CTA Button**: "Mulai Analisis" → Scroll to Upload

#### **3. 📤 UPLOAD SECTION (#upload-section)**
- **Upload Area**: Drag & Drop / Click to Upload
- **File Support**: JPG, PNG, PDF (Max: 10MB)
- **Preview Area**: 
  - Image preview
  - File info (name, size)
  - "Mulai Analisis" button

#### **4. ⏳ ANALYSIS PROGRESS (#analysis-progress)**
- **Step 1**: Analisis ML/DL (Random Forest & TensorFlow)
- **Step 2**: Ekstraksi Teks OCR (Tesseract)
- **Step 3**: Verifikasi Teks (Manual editing)
- **Step 4**: Analisis Detail 4 Model AI
- **Step 5**: Hasil & Rekomendasi

#### **5. 📊 RESULTS SECTION (#results-section)**
- **ML Analysis Card**: Confidence meter + status
- **OCR Text Card**: 
  - OCR quality notice
  - External OCR recommendations
  - Text editor with edit/save functionality
- **Text Analysis Card**: Confidence meter + reanalyze button
- **Detailed Analysis Card**: 4 AI models breakdown
- **Final Result Card**: 
  - Large status display
  - Threshold legend (< 40%, 40-80%, > 80%)
  - Action buttons (Download, New Analysis)

#### **6. 🔄 LOADING OVERLAY**
- **Spinner Animation**
- **Dynamic Loading Text**
- **Progress Indication**

---

## 🔗 **INTERNAL NAVIGATION FLOW**

### **📱 USER JOURNEY PATHS**

#### **Path 1: Standard Analysis Flow**
```
Landing → Upload → Analysis → Results → Action
```

#### **Path 2: Text Re-analysis Flow**
```
Results → Edit Text → Re-analyze → Updated Results
```

#### **Path 3: New Analysis Flow**
```
Results → New Analysis → Upload → Analysis → Results
```

---

## 🎛️ **INTERACTIVE ELEMENTS**

### **📤 Upload Interactions**
- **Drag & Drop**: File upload area
- **Click Upload**: File picker dialog
- **File Preview**: Image display + info
- **File Validation**: Format & size checking

### **📝 Text Editing**
- **Edit Button**: Enable text editing
- **Save Button**: Save edited text
- **Re-analyze Button**: Trigger new analysis with edited text

### **🔄 System Controls**
- **Refresh Status**: Update system status
- **Scroll to Upload**: Smooth scroll navigation
- **Download Report**: Generate analysis report
- **Reset Analysis**: Start new analysis

### **📊 Progress Tracking**
- **Step Indicators**: Visual progress steps
- **Progress Bars**: Individual step completion
- **Loading States**: Dynamic status updates

---

## 🎨 **VISUAL HIERARCHY**

### **🏗️ LAYOUT STRUCTURE**
```
Header (Fixed)
├── Logo + Navigation
│
Hero Section
├── Title + Description
├── System Status Grid
└── CTA Button
│
Upload Section
├── Upload Area
└── Preview Area
│
Analysis Progress
├── Step 1: ML/DL Analysis
├── Step 2: OCR Extraction
├── Step 3: Text Verification
├── Step 4: Detailed AI Analysis
└── Step 5: Final Results
│
Results Section
├── ML Analysis Card
├── OCR Text Card
├── Text Analysis Card
├── Detailed Analysis Card
└── Final Result Card
│
Loading Overlay (Modal)
└── Spinner + Status Text
```

### **🎯 CONTENT PRIORITY**
1. **Primary**: Upload & Analysis functionality
2. **Secondary**: Results display & interpretation
3. **Tertiary**: System status & navigation
4. **Supporting**: OCR recommendations & help text

---

## 📱 **RESPONSIVE BEHAVIOR**

### **💻 Desktop Layout**
- **Full navigation menu**
- **Grid-based results layout**
- **Side-by-side content arrangement**

### **📱 Mobile Layout**
- **Hamburger menu** (if implemented)
- **Stacked card layout**
- **Touch-optimized interactions**
- **Simplified upload interface**

---

## 🔧 **FUNCTIONAL COMPONENTS**

### **🎛️ Interactive Components**
- **File Upload Handler**
- **Progress Step Manager**
- **Text Editor Controller**
- **Results Display Manager**
- **System Status Monitor**

### **📊 Data Flow Components**
- **API Communication Layer**
- **State Management System**
- **Error Handling Framework**
- **Loading State Controller**

---

## 🚀 **FUTURE NAVIGATION ENHANCEMENTS**

### **📋 Suggested Additions**
1. **About Page**: Detailed information about the project
2. **How It Works**: Step-by-step explanation
3. **FAQ Section**: Common questions and answers
4. **Contact Page**: Support and feedback
5. **History Page**: Previous analysis results
6. **Settings Page**: User preferences and configurations

### **🔧 Technical Improvements**
1. **Breadcrumb Navigation**: Current location indicator
2. **Search Functionality**: Find specific features
3. **Keyboard Navigation**: Accessibility improvements
4. **Deep Linking**: Direct access to specific sections
5. **Progressive Web App**: Offline functionality

---

## 📊 **NAVIGATION ANALYTICS**

### **🎯 Key Metrics to Track**
- **Upload Completion Rate**
- **Analysis Success Rate**
- **Text Edit Usage**
- **Download Report Usage**
- **New Analysis Rate**
- **System Status Check Frequency**

### **🔍 User Behavior Insights**
- **Most Used Features**
- **Drop-off Points**
- **Error Frequency**
- **Mobile vs Desktop Usage**
- **Average Session Duration**
