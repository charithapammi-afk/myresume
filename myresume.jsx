import React, { useState, useEffect } from 'react';
import { Upload, Search, FileText, TrendingUp, Award, Brain, BarChart3, Download, Trash2, AlertCircle, CheckCircle2, Sparkles } from 'lucide-react';

export default function MLResumeScreener() {
  const [resumes, setResumes] = useState([]);
  const [jobDescription, setJobDescription] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [analytics, setAnalytics] = useState(null);
  const [selectedTab, setSelectedTab] = useState('upload');

  // ML Functions - TF-IDF and Cosine Similarity
  const calculateTFIDF = (documents) => {
    const vocabulary = new Set();
    const docTermFreqs = [];

    // Build vocabulary and term frequencies
    documents.forEach(doc => {
      const terms = doc.toLowerCase().match(/\b[a-z]{2,}\b/g) || [];
      const termFreq = {};
      terms.forEach(term => {
        vocabulary.add(term);
        termFreq[term] = (termFreq[term] || 0) + 1;
      });
      docTermFreqs.push(termFreq);
    });

    // Calculate IDF
    const vocabArray = Array.from(vocabulary);
    const idf = {};
    vocabArray.forEach(term => {
      const docsWithTerm = docTermFreqs.filter(tf => tf[term]).length;
      idf[term] = Math.log(documents.length / (docsWithTerm || 1));
    });

    // Calculate TF-IDF vectors
    return docTermFreqs.map(termFreq => {
      const vector = {};
      vocabArray.forEach(term => {
        const tf = termFreq[term] || 0;
        vector[term] = tf * (idf[term] || 0);
      });
      return vector;
    });
  };

  const cosineSimilarity = (vec1, vec2) => {
    const keys = new Set([...Object.keys(vec1), ...Object.keys(vec2)]);
    let dotProduct = 0;
    let mag1 = 0;
    let mag2 = 0;

    keys.forEach(key => {
      const v1 = vec1[key] || 0;
      const v2 = vec2[key] || 0;
      dotProduct += v1 * v2;
      mag1 += v1 * v1;
      mag2 += v2 * v2;
    });

    return dotProduct / (Math.sqrt(mag1) * Math.sqrt(mag2)) || 0;
  };

  // Extract skills using NLP patterns
  const extractSkills = (text) => {
    const skillsDatabase = [
      'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node',
      'django', 'flask', 'spring', 'sql', 'mongodb', 'postgresql', 'mysql',
      'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
      'html', 'css', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go',
      'machine learning', 'deep learning', 'ai', 'data science', 'nlp',
      'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
      'agile', 'scrum', 'devops', 'ci/cd', 'rest api', 'graphql',
      'redux', 'next.js', 'express', 'fastapi', 'microservices'
    ];

    const lowerText = text.toLowerCase();
    const foundSkills = skillsDatabase.filter(skill => 
      lowerText.includes(skill)
    );

    return [...new Set(foundSkills)];
  };

  // Extract experience level
  const extractExperience = (text) => {
    const expPatterns = [
      /(\d+)\+?\s*years?/gi,
      /(\d+)\+?\s*yrs?/gi,
      /experience[:\s]*(\d+)/gi
    ];

    let maxYears = 0;
    expPatterns.forEach(pattern => {
      const matches = text.matchAll(pattern);
      for (const match of matches) {
        const years = parseInt(match[1]);
        if (years > maxYears) maxYears = years;
      }
    });

    if (maxYears === 0) return 'Fresher';
    if (maxYears <= 2) return 'Junior (0-2 years)';
    if (maxYears <= 5) return 'Mid-level (3-5 years)';
    return `Senior (${maxYears}+ years)`;
  };

  // PDF text extraction
  const extractTextFromPDF = async (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = async (e) => {
        try {
          const typedArray = new Uint8Array(e.target.result);
          const pdfjsLib = window['pdfjs-dist/build/pdf'];
          pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
          
          const pdf = await pdfjsLib.getDocument(typedArray).promise;
          let fullText = '';
          
          for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const textContent = await page.getTextContent();
            const pageText = textContent.items.map(item => item.str).join(' ');
            fullText += pageText + ' ';
          }
          
          resolve(fullText);
        } catch (error) {
          reject(error);
        }
      };
      
      reader.onerror = reject;
      reader.readAsArrayBuffer(file);
    });
  };

  // Handle file upload
  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;
    
    setIsProcessing(true);
    setUploadStatus(`Processing ${files.length} resumes with AI...`);
    
    const processedResumes = [];
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      
      if (file.type === 'application/pdf') {
        try {
          setUploadStatus(`AI Processing ${i + 1}/${files.length}: ${file.name}`);
          const text = await extractTextFromPDF(file);
          const skills = extractSkills(text);
          const experience = extractExperience(text);
          
          processedResumes.push({
            id: Date.now() + i,
            name: file.name.replace('.pdf', ''),
            fileName: file.name,
            content: text,
            skills: skills,
            experience: experience,
            matchScore: 0,
            uploadedAt: new Date().toLocaleString()
          });
        } catch (error) {
          console.error(`Error processing ${file.name}:`, error);
        }
      }
    }
    
    setResumes(prev => [...prev, ...processedResumes]);
    setIsProcessing(false);
    setUploadStatus(`✅ AI processed ${processedResumes.length} resumes successfully!`);
    generateAnalytics([...resumes, ...processedResumes]);
    
    setTimeout(() => setUploadStatus(''), 3000);
  };

  // ML-powered search
  const performMLSearch = () => {
    if (!searchQuery.trim() && !jobDescription.trim()) {
      setResults([]);
      return;
    }

    const query = searchQuery || jobDescription;
    const documents = resumes.map(r => r.content);
    documents.push(query);

    const tfidfVectors = calculateTFIDF(documents);
    const queryVector = tfidfVectors[tfidfVectors.length - 1];

    const scoredResumes = resumes.map((resume, idx) => {
      const resumeVector = tfidfVectors[idx];
      const similarity = cosineSimilarity(queryVector, resumeVector);
      const matchScore = Math.round(similarity * 100);

      // Bonus scoring for skill matches
      const querySkills = extractSkills(query);
      const matchedSkills = resume.skills.filter(skill => 
        querySkills.includes(skill)
      );
      const skillBonus = (matchedSkills.length / Math.max(querySkills.length, 1)) * 20;

      return {
        ...resume,
        matchScore: Math.min(100, matchScore + skillBonus),
        matchedSkills: matchedSkills
      };
    });

    const ranked = scoredResumes
      .filter(r => r.matchScore > 10)
      .sort((a, b) => b.matchScore - a.matchScore);

    setResults(ranked);
    setSelectedTab('results');
  };

  // Generate analytics
  const generateAnalytics = (resumeList) => {
    const allSkills = {};
    const experienceLevels = {
      'Fresher': 0,
      'Junior (0-2 years)': 0,
      'Mid-level (3-5 years)': 0,
      'Senior': 0
    };

    resumeList.forEach(resume => {
      resume.skills.forEach(skill => {
        allSkills[skill] = (allSkills[skill] || 0) + 1;
      });

      if (resume.experience.includes('Fresher')) experienceLevels['Fresher']++;
      else if (resume.experience.includes('Junior')) experienceLevels['Junior (0-2 years)']++;
      else if (resume.experience.includes('Mid-level')) experienceLevels['Mid-level (3-5 years)']++;
      else experienceLevels['Senior']++;
    });

    const topSkills = Object.entries(allSkills)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);

    setAnalytics({
      totalResumes: resumeList.length,
      topSkills: topSkills,
      experienceLevels: experienceLevels,
      avgSkillsPerResume: (Object.values(allSkills).reduce((a, b) => a + b, 0) / resumeList.length).toFixed(1)
    });
  };

  useEffect(() => {
    if (resumes.length > 0) {
      generateAnalytics(resumes);
    }
  }, [resumes]);

  const exportResults = () => {
    const csvContent = [
      ['Rank', 'Candidate Name', 'Match Score', 'Experience', 'Skills', 'Matched Skills'],
      ...results.map((r, idx) => [
        idx + 1,
        r.name,
        `${r.matchScore}%`,
        r.experience,
        r.skills.join('; '),
        r.matchedSkills.join('; ')
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `AI_Resume_Results_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-xl shadow-xl p-8 mb-6 border-t-4 border-purple-600">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center mb-2">
                <Brain className="text-purple-600 mr-3" size={40} />
                <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                  AI-Powered Resume Screening System
                </h1>
              </div>
              <p className="text-gray-600 ml-14">Machine Learning • NLP • Intelligent Matching • Analytics</p>
            </div>
            <Sparkles className="text-yellow-500" size={48} />
          </div>
        </div>

        {/* Tabs */}
        <div className="bg-white rounded-xl shadow-lg p-2 mb-6 flex gap-2">
          <button
            onClick={() => setSelectedTab('upload')}
            className={`flex-1 py-3 px-4 rounded-lg font-semibold transition flex items-center justify-center ${
              selectedTab === 'upload' ? 'bg-purple-600 text-white' : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            <Upload size={20} className="mr-2" />
            Upload Resumes
          </button>
          <button
            onClick={() => setSelectedTab('search')}
            className={`flex-1 py-3 px-4 rounded-lg font-semibold transition flex items-center justify-center ${
              selectedTab === 'search' ? 'bg-purple-600 text-white' : 'text-gray-600 hover:bg-gray-100'
            }`}
            disabled={resumes.length === 0}
          >
            <Search size={20} className="mr-2" />
            AI Search
          </button>
          <button
            onClick={() => setSelectedTab('results')}
            className={`flex-1 py-3 px-4 rounded-lg font-semibold transition flex items-center justify-center ${
              selectedTab === 'results' ? 'bg-purple-600 text-white' : 'text-gray-600 hover:bg-gray-100'
            }`}
            disabled={results.length === 0}
          >
            <Award size={20} className="mr-2" />
            Results ({results.length})
          </button>
          <button
            onClick={() => setSelectedTab('analytics')}
            className={`flex-1 py-3 px-4 rounded-lg font-semibold transition flex items-center justify-center ${
              selectedTab === 'analytics' ? 'bg-purple-600 text-white' : 'text-gray-600 hover:bg-gray-100'
            }`}
            disabled={!analytics}
          >
            <BarChart3 size={20} className="mr-2" />
            Analytics
          </button>
        </div>

        {/* Upload Tab */}
        {selectedTab === 'upload' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
                <Upload className="mr-3 text-purple-600" size={28} />
                Upload Resumes for AI Analysis
              </h2>
              
              <label className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-purple-300 rounded-xl cursor-pointer hover:border-purple-500 transition bg-purple-50 hover:bg-purple-100">
                <div className="flex flex-col items-center justify-center">
                  <Brain className="mb-3 text-purple-500" size={48} />
                  <p className="mb-2 text-lg text-gray-700 font-semibold">AI-Powered Processing</p>
                  <p className="text-sm text-gray-600">Upload PDF resumes for intelligent analysis</p>
                  <p className="text-xs text-gray-500 mt-2">Auto skill extraction • Experience detection • Smart matching</p>
                </div>
                <input
                  type="file"
                  className="hidden"
                  multiple
                  accept=".pdf"
                  onChange={handleFileUpload}
                  disabled={isProcessing}
                />
              </label>
              
              {uploadStatus && (
                <div className={`mt-4 p-4 rounded-lg flex items-center ${isProcessing ? 'bg-blue-50 border border-blue-200' : 'bg-green-50 border border-green-200'}`}>
                  {isProcessing ? <AlertCircle className="text-blue-500 mr-3" size={24} /> : <CheckCircle2 className="text-green-500 mr-3" size={24} />}
                  <span className="text-gray-700 font-medium">{uploadStatus}</span>
                </div>
              )}
              
              <div className="mt-6 grid grid-cols-3 gap-4">
                <div className="bg-gradient-to-br from-purple-500 to-purple-600 text-white p-4 rounded-lg">
                  <div className="text-3xl font-bold">{resumes.length}</div>
                  <div className="text-sm opacity-90">Total Resumes</div>
                </div>
                <div className="bg-gradient-to-br from-blue-500 to-blue-600 text-white p-4 rounded-lg">
                  <div className="text-3xl font-bold">{analytics?.avgSkillsPerResume || 0}</div>
                  <div className="text-sm opacity-90">Avg Skills/Resume</div>
                </div>
                <div className="bg-gradient-to-br from-indigo-500 to-indigo-600 text-white p-4 rounded-lg">
                  <div className="text-3xl font-bold">{analytics?.topSkills.length || 0}</div>
                  <div className="text-sm opacity-90">Unique Skills</div>
                </div>
              </div>
            </div>

            {resumes.length > 0 && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-xl font-bold text-gray-800">Processed Resumes</h3>
                  <button
                    onClick={() => { setResumes([]); setResults([]); setAnalytics(null); }}
                    className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition flex items-center"
                  >
                    <Trash2 size={18} className="mr-2" />
                    Clear All
                  </button>
                </div>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {resumes.map(resume => (
                    <div key={resume.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition">
                      <div className="flex justify-between items-start">
                        <div className="flex-1">
                          <h4 className="font-semibold text-gray-800 text-lg">{resume.name}</h4>
                          <p className="text-sm text-gray-600 mt-1">{resume.experience}</p>
                          <div className="flex flex-wrap gap-2 mt-2">
                            {resume.skills.slice(0, 5).map((skill, idx) => (
                              <span key={idx} className="px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded-full">
                                {skill}
                              </span>
                            ))}
                            {resume.skills.length > 5 && (
                              <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full">
                                +{resume.skills.length - 5} more
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Search Tab */}
        {selectedTab === 'search' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
                <Brain className="mr-3 text-purple-600" size={28} />
                AI-Powered Intelligent Search
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Job Description (Optional)</label>
                  <textarea
                    value={jobDescription}
                    onChange={(e) => setJobDescription(e.target.value)}
                    placeholder="Paste job description here for AI matching..."
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none h-32"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Or Search by Skills/Keywords</label>
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && performMLSearch()}
                    placeholder="E.g., python machine learning, react developer, java spring boot..."
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none"
                  />
                </div>
                
                <button
                  onClick={performMLSearch}
                  className="w-full py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:from-purple-700 hover:to-blue-700 transition font-bold text-lg flex items-center justify-center"
                >
                  <Sparkles className="mr-2" size={24} />
                  Run AI Analysis
                </button>
              </div>
              
              <div className="mt-6 bg-purple-50 border border-purple-200 rounded-lg p-4">
                <p className="text-sm text-gray-700">
                  <strong>AI Features:</strong> Semantic matching • Skill extraction • Experience detection • 
                  TF-IDF analysis • Cosine similarity • Intelligent ranking
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Results Tab */}
        {selectedTab === 'results' && results.length > 0 && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-gray-800 flex items-center">
                  <Award className="mr-3 text-purple-600" size={28} />
                  Top Candidates (Ranked by AI)
                </h2>
                <button
                  onClick={exportResults}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition flex items-center"
                >
                  <Download size={18} className="mr-2" />
                  Export Results
                </button>
              </div>
              
              <div className="space-y-4">
                {results.map((resume, idx) => (
                  <div key={resume.id} className="border-2 border-gray-200 rounded-xl p-6 hover:shadow-lg transition bg-gradient-to-r from-white to-purple-50">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start flex-1">
                        <div className="flex flex-col items-center mr-4">
                          <div className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-lg ${
                            idx === 0 ? 'bg-yellow-400 text-yellow-900' :
                            idx === 1 ? 'bg-gray-300 text-gray-700' :
                            idx === 2 ? 'bg-orange-300 text-orange-900' :
                            'bg-purple-100 text-purple-700'
                          }`}>
                            #{idx + 1}
                          </div>
                        </div>
                        
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-2">
                            <h3 className="text-xl font-bold text-gray-800">{resume.name}</h3>
                            <div className="flex items-center">
                              <TrendingUp className="text-green-500 mr-2" size={20} />
                              <span className="text-2xl font-bold text-green-600">{resume.matchScore}%</span>
                            </div>
                          </div>
                          
                          <p className="text-gray-600 mb-3">{resume.experience}</p>
                          
                          {resume.matchedSkills.length > 0 && (
                            <div className="mb-3">
                              <p className="text-sm font-semibold text-gray-700 mb-2">Matched Skills:</p>
                              <div className="flex flex-wrap gap-2">
                                {resume.matchedSkills.map((skill, idx) => (
                                  <span key={idx} className="px-3 py-1 bg-green-100 text-green-700 text-sm rounded-full font-medium">
                                    ✓ {skill}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          <div>
                            <p className="text-sm font-semibold text-gray-700 mb-2">All Skills:</p>
                            <div className="flex flex-wrap gap-2">
                              {resume.skills.map((skill, idx) => (
                                <span key={idx} className="px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded-full">
                                  {skill}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Analytics Tab */}
        {selectedTab === 'analytics' && analytics && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                  <BarChart3 className="mr-2 text-purple-600" size={24} />
                  Top Skills in Database
                </h3>
                <div className="space-y-3">
                  {analytics.topSkills.map(([skill, count], idx) => (
                    <div key={idx}>
                      <div className="flex justify-between mb-1">
                        <span className="font-medium text-gray-700">{skill}</span>
                        <span className="text-purple-600 font-bold">{count} candidates</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full transition-all"
                          style={{ width: `${(count / analytics.totalResumes) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                  <TrendingUp className="mr-2 text-purple-600" size={24} />
                  Experience Level Distribution
                </h3>
                <div className="space-y-4">
                  {Object.entries(analytics.experienceLevels).map(([level, count]) => (
                    <div key={level}>
                      <div className="flex justify-between mb-1">
                        <span className="font-medium text-gray-700">{level}</span>
                        <span className="text-blue-600 font-bold">{count}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div
                          className="bg-gradient-to-r from-blue-500 to-indigo-500 h-3 rounded-full transition-all"
                          style={{ width: `${(count / analytics.totalResumes) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-800 mb-4">Database Overview</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gradient-to-br from-green-500 to-green-600 text-white p-4 rounded-lg">
                  <div className="text-3xl font-bold">{analytics.totalResumes}</div>
                  <div className="text-sm opacity-90">Total Candidates</div>
                </div>
                <div className="bg-gradient-to-br from-orange-500 to-orange-600 text-white p-4 rounded-lg">
                  <div className="text-3xl font-bold">{analytics.topSkills.length}</div>
                  <div className="text-sm opacity-90">Unique Skills</div>
                </div>
                <div className="bg-gradient-to-br from-pink-500 to-pink-600 text-white p-4 rounded-lg">
                  <div className="text-3xl font-bold">{analytics.avgSkillsPerResume}</div>
                  <div className="text-sm opacity-90">Avg Skills/Resume</div>
                </div>
                <div className="bg-gradient-to-br from-teal-500 to-teal-600 text-white p-4 rounded-lg">
                  <div className="text-3xl font-bold">{results.length}</div>
                  <div className="text-sm opacity-90">Matched Results</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Empty State for Results */}
        {selectedTab === 'results' && results.length === 0 && (
          <div className="bg-white rounded-xl shadow-lg p-12 text-center">
            <Award className="mx-auto text-gray-300 mb-4" size={64} />
            <h3 className="text-2xl font-bold text-gray-800 mb-2">No Results Yet</h3>
            <p className="text-gray-600 mb-6">Run an AI search to find matching candidates</p>
            <button
              onClick={() => setSelectedTab('search')}
              className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition"
            >
              Go to AI Search
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
