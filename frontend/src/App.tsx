import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  Sparkles, 
  Zap, 
  Download, 
  Copy, 
  Image as ImageIcon,
  Settings2,
  AlertCircle,
  Loader2,
  Menu,
  LayoutGrid,
  Clock,
  CreditCard,
  User,
  X,
  MoreHorizontal
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// --- Utility ---
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Configuration ---
const API_URL = 'http://localhost:5000';

// --- Types ---
interface JobResponse {
  job_id: string;
  status: 'queued';
}

interface StatusResponse {
  status: 'queued' | 'processing' | 'completed' | 'failed';
  image_url?: string;
  error?: string;
  progress?: number; 
}

// Matches Python DB Structure
interface JobItem {
  id: string;
  prompt: string;
  negative_prompt: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  created_at: string;
  image_filename?: string;
}

function App() {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  
  // Tabs & Data
  const [activeTab, setActiveTab] = useState<'create' | 'gallery' | 'history'>('create');
  const [galleryItems, setGalleryItems] = useState<JobItem[]>([]);
  const [historyItems, setHistoryItems] = useState<JobItem[]>([]);
  const [isLoadingData, setIsLoadingData] = useState(false);

  // Generation State
  const [status, setStatus] = useState<'idle' | 'queued' | 'processing' | 'completed' | 'failed'>('idle');
  const [jobId, setJobId] = useState<string | null>(null);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingMessage, setLoadingMessage] = useState('Queued...');
  const [progress, setProgress] = useState(0);

  const pollIntervalRef = useRef<number | null>(null);

  // --- Effects ---

  // Fetch data when tabs change
  useEffect(() => {
    if (activeTab === 'gallery') fetchGallery();
    if (activeTab === 'history') fetchHistory();
  }, [activeTab]);

  const fetchGallery = async () => {
    setIsLoadingData(true);
    try {
      const res = await axios.get<JobItem[]>(`${API_URL}/gallery`);
      setGalleryItems(res.data);
    } catch (e) {
      console.error("Failed to fetch gallery", e);
    } finally {
      setIsLoadingData(false);
    }
  };

  const fetchHistory = async () => {
    setIsLoadingData(true);
    try {
      const res = await axios.get<JobItem[]>(`${API_URL}/history`);
      setHistoryItems(res.data);
    } catch (e) {
      console.error("Failed to fetch history", e);
    } finally {
      setIsLoadingData(false);
    }
  };

  // --- Generation Logic ---

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setStatus('queued');
    setGeneratedImage(null);
    setError(null);
    setProgress(0);
    setLoadingMessage('Allocating GPU...');

    try {
      const response = await axios.post<JobResponse>(`${API_URL}/generate`, {
        prompt,
        negative_prompt: negativePrompt
      });
      
      const { job_id } = response.data;
      setJobId(job_id);
      startPolling(job_id);
    } catch (err: any) {
      console.error(err);
      setStatus('failed');
      setError(err.response?.data?.error || 'Failed to start generation job.');
    }
  };

  const startPolling = (id: string) => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);

    pollIntervalRef.current = window.setInterval(async () => {
      try {
        const res = await axios.get<StatusResponse>(`${API_URL}/status/${id}`);
        const data = res.data;

        if (data.status === 'processing') {
          setStatus('processing');
          setLoadingMessage(() => {
            const msgs = ['Dreaming...', 'Diffusing...', 'Applying LoRA...', 'Adding details...'];
            return msgs[Math.floor(Math.random() * msgs.length)];
          });
        } else if (data.status === 'completed' && data.image_url) {
          setPrompt(''); 
          setStatus('completed');
          setGeneratedImage(data.image_url);
          stopPolling();
        } else if (data.status === 'failed') {
          setStatus('failed');
          setError(data.error || 'Generation failed on server.');
          stopPolling();
        }
      } catch (err) {
        console.warn('Status check failed', err);
      }
    }, 2000);
  };

  const stopPolling = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  };

  useEffect(() => {
    return () => stopPolling();
  }, []);

  // --- Helpers ---

  const copyPrompt = (text: string = prompt) => {
    navigator.clipboard.writeText(text);
  };

  const reusePrompt = (item: JobItem) => {
    setPrompt(item.prompt);
    setNegativePrompt(item.negative_prompt || '');
    setActiveTab('create');
  };

  const getImageUrl = (filename: string) => `${API_URL}/static/generated/${filename}`;

  // --- Render ---

  return (
    <div className="min-h-screen w-full bg-zinc-950 text-white selection:bg-purple-500/30 overflow-x-hidden font-sans flex flex-col">
      {/* Background Decor */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>
        <div className="absolute top-0 left-0 w-full h-[50vh] bg-gradient-to-b from-purple-900/10 to-transparent pointer-events-none"></div>
      </div>

      {/* Navbar */}
      <header className="sticky top-0 z-50 w-full border-b border-white/5 bg-zinc-950/80 backdrop-blur-xl">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => setActiveTab('create')}>
            <div className="relative group">
              <div className="absolute -inset-1 bg-gradient-to-r from-purple-600 to-cyan-500 rounded-lg blur opacity-40 group-hover:opacity-75 transition duration-500"></div>
              <div className="relative p-2 bg-zinc-900 rounded-lg ring-1 ring-white/10">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
            </div>
            <span className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-zinc-200 to-zinc-400 bg-clip-text text-transparent">
              LoRA
            </span>
          </div>

          <nav className="hidden md:flex items-center gap-1 bg-white/5 p-1.5 rounded-full border border-white/5 shadow-inner shadow-black/20">
            {[
              { id: 'create', icon: Sparkles, label: 'Create', color: 'text-purple-400' },
              { id: 'gallery', icon: LayoutGrid, label: 'Gallery', color: 'text-cyan-400' },
              { id: 'history', icon: Clock, label: 'History', color: 'text-amber-400' }
            ].map((tab) => (
              <button 
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all",
                  activeTab === tab.id 
                    ? "bg-zinc-800 text-white shadow-sm ring-1 ring-white/10" 
                    : "text-zinc-400 hover:text-white hover:bg-white/5"
                )}
              >
                <tab.icon className={cn("w-4 h-4", activeTab === tab.id && tab.color)} />
                {tab.label}
              </button>
            ))}
          </nav>

          <div className="flex items-center gap-4">
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-xs font-medium text-zinc-300">
              <CreditCard className="w-3.5 h-3.5 text-cyan-500" />
              <span>Free Tier</span>
            </div>
            <button 
              className="md:hidden p-2 text-zinc-400 hover:text-white"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 relative z-10 w-full max-w-6xl mx-auto px-6 py-8 md:py-12 flex flex-col">
        
        {/* --- CREATE TAB --- */}
        {activeTab === 'create' && (
          <div className="grid grid-cols-1 md:grid-cols-12 gap-8 h-full animate-in fade-in zoom-in-95 duration-300">
            <section className="md:col-span-5 flex flex-col gap-6 order-2 md:order-1 h-full">
              <div className="bg-zinc-900/60 backdrop-blur-xl border border-white/10 rounded-2xl p-5 shadow-2xl ring-1 ring-white/5 flex flex-col gap-4">
                <form onSubmit={handleGenerate} className="flex flex-col gap-4">
                  <div className="space-y-2">
                    <label className="text-sm text-zinc-400 font-medium">Prompt</label>
                    <textarea
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder="Describe your imagination..."
                      className="w-full h-32 bg-black/40 text-base text-white p-4 rounded-xl border border-white/10 focus:border-purple-500/50 focus:outline-none resize-none"
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={!prompt.trim() || status === 'queued' || status === 'processing'}
                    className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-purple-600 to-cyan-600 text-white py-3 rounded-xl font-bold hover:scale-[1.02] disabled:opacity-50 disabled:hover:scale-100 transition-all shadow-lg shadow-purple-500/20"
                  >
                    {status === 'queued' || status === 'processing' ? <Loader2 className="w-5 h-5 animate-spin" /> : <Sparkles className="w-5 h-5" />}
                    Generate
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex items-center gap-2 text-xs font-medium text-zinc-500 hover:text-zinc-300 justify-center"
                  >
                    <Settings2 className="w-3.5 h-3.5" />
                    {showAdvanced ? 'Hide Advanced' : 'Show Advanced'}
                  </button>
                  {showAdvanced && (
                    <input 
                      type="text"
                      value={negativePrompt}
                      onChange={(e) => setNegativePrompt(e.target.value)}
                      placeholder="Negative Prompt (e.g. blurry, ugly)"
                      className="w-full bg-zinc-800/50 rounded-lg px-3 py-2 text-sm text-zinc-300 border border-white/5"
                    />
                  )}
                </form>
              </div>
            </section>

            <section className="md:col-span-7 order-1 md:order-2">
              <div className={cn(
                "relative w-full h-full min-h-[400px] rounded-3xl overflow-hidden shadow-2xl transition-all border border-white/10",
                status === 'idle' ? "bg-zinc-900/40 border-dashed" : "bg-black"
              )}>
                {status === 'idle' && (
                  <div className="flex flex-col items-center justify-center h-full text-zinc-500 gap-4">
                    <ImageIcon className="w-16 h-16 opacity-20" />
                    <p>Enter a prompt to begin.</p>
                  </div>
                )}
                
                {(status === 'queued' || status === 'processing') && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 backdrop-blur-md">
                    <Zap className="w-10 h-10 text-cyan-400 animate-pulse mb-4" />
                    <h3 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-cyan-400">{loadingMessage}</h3>
                  </div>
                )}

                {status === 'completed' && generatedImage && (
                  <div className="group relative w-full h-full flex items-center justify-center bg-[#1a1a1a]">
                    <img src={generatedImage} alt={prompt} className="max-w-full max-h-full object-contain" />
                    <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/90 p-6 translate-y-full group-hover:translate-y-0 transition-transform flex justify-end gap-3">
                      <button onClick={() => copyPrompt()} className="p-3 rounded-xl bg-white/10 hover:bg-white/20 text-white"><Copy className="w-5 h-5" /></button>
                      <a href={generatedImage} download={`lora-${jobId}.png`} target="_blank" rel="noreferrer" className="flex items-center gap-2 bg-white text-black font-bold py-3 px-6 rounded-xl hover:bg-zinc-200">
                        <Download className="w-4 h-4" /> Save
                      </a>
                    </div>
                  </div>
                )}

                {status === 'failed' && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center text-red-400">
                    <AlertCircle className="w-12 h-12 mb-4" />
                    <p className="text-center px-4">{error}</p>
                    <button onClick={() => setStatus('idle')} className="mt-4 px-6 py-2 bg-red-500/10 rounded-lg">Reset</button>
                  </div>
                )}
              </div>
            </section>
          </div>
        )}

        {/* --- GALLERY TAB --- */}
        {activeTab === 'gallery' && (
          <div className="animate-in fade-in zoom-in-95 duration-300">
            <h2 className="text-2xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-cyan-400">Your Gallery</h2>
            
            {isLoadingData ? (
               <div className="flex justify-center py-20"><Loader2 className="w-8 h-8 animate-spin text-purple-500" /></div>
            ) : galleryItems.length === 0 ? (
               <div className="text-center py-20 text-zinc-500">No images generated yet.</div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {galleryItems.map((item) => (
                  <div key={item.id} className="group relative aspect-square rounded-xl overflow-hidden bg-zinc-900 border border-white/5">
                    <img 
                      src={getImageUrl(item.image_filename!)} 
                      alt={item.prompt} 
                      className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                    />
                    <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                       <a href={getImageUrl(item.image_filename!)} download={`lora-${item.id}.png`} target="_blank" rel="noreferrer" className="p-3 rounded-full bg-white/20 hover:bg-white/30 text-white backdrop-blur-md"><Download className="w-5 h-5" /></a>
                       <button onClick={() => reusePrompt(item)} className="p-3 rounded-full bg-white/20 hover:bg-white/30 text-white backdrop-blur-md"><Sparkles className="w-5 h-5" /></button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* --- HISTORY TAB --- */}
        {activeTab === 'history' && (
          <div className="max-w-3xl mx-auto w-full animate-in fade-in zoom-in-95 duration-300">
             <h2 className="text-2xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-amber-200 to-orange-400">Prompt Logs</h2>
             
             {isLoadingData ? (
               <div className="flex justify-center py-20"><Loader2 className="w-8 h-8 animate-spin text-amber-500" /></div>
             ) : historyItems.length === 0 ? (
               <div className="text-center py-20 text-zinc-500">No history found.</div>
             ) : (
               <div className="space-y-3">
                 {historyItems.map((item) => (
                   <div key={item.id} className="bg-zinc-900/50 border border-white/5 p-4 rounded-xl flex flex-col gap-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-zinc-500 font-mono">{new Date(item.created_at).toLocaleString()}</span>
                        <span className={cn("text-[10px] uppercase font-bold px-2 py-0.5 rounded-full", 
                            item.status === 'completed' ? "bg-green-500/10 text-green-400" : 
                            item.status === 'failed' ? "bg-red-500/10 text-red-400" : "bg-zinc-500/10 text-zinc-400"
                        )}>
                          {item.status}
                        </span>
                      </div>
                      <div className="flex gap-4">
                        <p className="flex-1 text-sm text-zinc-300 font-mono bg-black/30 p-3 rounded-lg overflow-x-auto whitespace-pre-wrap">{item.prompt}</p>
                        {item.image_filename && (
                          <img src={getImageUrl(item.image_filename)} className="w-16 h-16 rounded-md object-cover border border-white/10" alt="thumbnail" />
                        )}
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        <button onClick={() => reusePrompt(item)} className="text-xs flex items-center gap-1 text-purple-400 hover:text-purple-300"><Sparkles className="w-3 h-3" /> Reuse</button>
                        <button onClick={() => copyPrompt(item.prompt)} className="text-xs flex items-center gap-1 text-zinc-500 hover:text-zinc-300"><Copy className="w-3 h-3" /> Copy</button>
                      </div>
                   </div>
                 ))}
               </div>
             )}
          </div>
        )}

      </main>
    </div>
  );
}

export default App;