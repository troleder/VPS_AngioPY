import React, { useState, useEffect } from 'react';
import { 
  signInWithEmailAndPassword, 
  onAuthStateChanged, 
  signOut 
} from 'firebase/auth';
import { auth } from './firebase';
import { 
  LayoutDashboard, 
  Building2, 
  UserCheck, 
  HardDrive, 
  AlertTriangle, 
  BarChart3, 
  UserPlus, 
  Upload, 
  LogOut, 
  RefreshCw, 
  CheckCircle2, 
  AlertCircle, 
  Loader2, 
  User,
  FolderSync,
  ChevronRight,
  UserCheck2,
  Lock,
  Database,
  Search,
  Trash2
} from 'lucide-react';

const API_BASE = '/api';

export default function App() {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('admin_token') || null);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [loading, setLoading] = useState(false);
  const [authLoading, setAuthLoading] = useState(true);
  
  // Auth state inputs
  const [emailInput, setEmailInput] = useState('');
  const [passwordInput, setPasswordInput] = useState('');
  const [authError, setAuthError] = useState(null);

  // Monitor Firebase Auth changes
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      if (firebaseUser) {
        // Verify user email matches admin criteria
        const email = firebaseUser.email.toLowerCase();
        const isAdmin = email === 'tomaszroleder@gmail.com' || email.startsWith('tomaszroleder@');
        if (isAdmin) {
          const userToken = await firebaseUser.getIdToken();
          localStorage.setItem('admin_token', userToken);
          setToken(userToken);
          setUser({
            email: firebaseUser.email,
            name: firebaseUser.displayName || firebaseUser.email.split('@')[0],
            username: firebaseUser.email.split('@')[0]
          });
        } else {
          setAuthError('Konto nie posiada uprawnień administratora.');
          await signOut(auth);
          handleLogout();
        }
      } else {
        handleLogout();
      }
      setAuthLoading(false);
    });
    return unsubscribe;
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('admin_token');
    setToken(null);
    setUser(null);
  };

  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    setAuthError(null);
    setLoading(true);
    try {
      await signInWithEmailAndPassword(auth, emailInput, passwordInput);
    } catch (err) {
      console.error(err);
      setAuthError('Błędny email lub hasło.');
    } finally {
      setLoading(false);
    }
  };

  const triggerSignOut = async () => {
    await signOut(auth);
    handleLogout();
  };

  if (authLoading) {
    return (
      <div style={{ display: 'flex', flex: 1, height: '100vh', justifyContent: 'center', alignItems: 'center', backgroundColor: '#0b0c10' }}>
        <Loader2 className="spin" size={48} color="#00ff00" />
      </div>
    );
  }

  if (!token || !user) {
    return (
      <div className="login-container" style={{ display: 'flex', flex: 1, minHeight: '100vh', justifyContent: 'center', alignItems: 'center', padding: '20px' }}>
        <div className="glass-card fade-in" style={{ width: '100%', maxWidth: '400px', border: '1px solid rgba(0, 255, 0, 0.2)' }}>
          <div style={{ textAlign: 'center', marginBottom: '24px' }}>
            <div style={{ display: 'inline-flex', padding: '12px', borderRadius: '50%', backgroundColor: 'rgba(0,255,0,0.1)', color: '#00ff00', marginBottom: '12px' }}>
              <Lock size={32} />
            </div>
            <h2 style={{ fontSize: '24px', fontWeight: '700', letterSpacing: '-0.5px', color: '#00ff00' }}>👑 Admin Panel</h2>
            <p style={{ fontSize: '14px', color: '#888', marginTop: '4px' }}>Zaloguj się kontem administratora AngioPy</p>
          </div>
          
          <form onSubmit={handleLoginSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {authError && (
              <div style={{ display: 'flex', gap: '8px', padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)', color: '#ef4444', fontSize: '14px' }}>
                <AlertCircle size={18} style={{ flexShrink: 0 }} />
                <span>{authError}</span>
              </div>
            )}
            
            <div>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '6px', color: '#aaa' }}>Email</label>
              <input 
                type="email" 
                value={emailInput} 
                onChange={e => setEmailInput(e.target.value)} 
                required 
                placeholder="np. admin@angiopy.tech"
              />
            </div>
            
            <div>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '6px', color: '#aaa' }}>Hasło</label>
              <input 
                type="password" 
                value={passwordInput} 
                onChange={e => setPasswordInput(e.target.value)} 
                required 
                placeholder="••••••"
              />
            </div>
            
            <button type="submit" className="btn-primary" disabled={loading} style={{ marginTop: '8px' }}>
              {loading ? <Loader2 className="spin" size={18} /> : 'Zaloguj się'}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flex: 1, minHeight: '100vh', width: '100%' }}>
      {/* Sidebar Navigation */}
      <aside style={{ width: '260px', borderRight: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', backgroundColor: 'rgba(31, 40, 51, 0.4)', flexShrink: 0 }}>
        <div style={{ padding: '24px', borderBottom: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ color: '#00ff00' }}>
            <Building2 size={28} />
          </div>
          <div>
            <h1 style={{ fontSize: '18px', fontWeight: '700', color: '#00ff00', letterSpacing: '-0.5px' }}>AngioPy</h1>
            <span style={{ fontSize: '11px', color: '#888', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: '600' }}>Admin Console</span>
          </div>
        </div>

        <nav style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '4px', flex: 1, overflowY: 'auto' }}>
          <TabButton id="dashboard" label="Dashboard" icon={<LayoutDashboard size={18} />} activeTab={activeTab} setActiveTab={setActiveTab} />
          <TabButton id="sites" label="Przypisz Ośrodek" icon={<Building2 size={18} />} activeTab={activeTab} setActiveTab={setActiveTab} />
          <TabButton id="patients" label="Przypisz Pacjenta" icon={<UserCheck size={18} />} activeTab={activeTab} setActiveTab={setActiveTab} />
          <TabButton id="caching" label="Pobieranie plików (Cache)" icon={<HardDrive size={18} />} activeTab={activeTab} setActiveTab={setActiveTab} />
          <TabButton id="cache_viewer" label="Stan Cache VPS" icon={<Database size={18} />} activeTab={activeTab} setActiveTab={setActiveTab} />
          <TabButton id="unassigned" label="Odpisane przypadki" icon={<AlertTriangle size={18} />} activeTab={activeTab} setActiveTab={setActiveTab} />
          <TabButton id="progress" label="Postęp analityków" icon={<BarChart3 size={18} />} activeTab={activeTab} setActiveTab={setActiveTab} />
          <TabButton id="analysts" label="Dodaj analityka" icon={<UserPlus size={18} />} activeTab={activeTab} setActiveTab={setActiveTab} />
          <TabButton id="import" label="Import PDF" icon={<Upload size={18} />} activeTab={activeTab} setActiveTab={setActiveTab} />
        </nav>

        <div style={{ padding: '16px', borderTop: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ width: '36px', height: '36px', borderRadius: '50%', backgroundColor: 'rgba(255, 255, 255, 0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#aaa', flexShrink: 0 }}>
              <User size={18} style={{ marginLeft: '9px' }} />
            </div>
            <div style={{ overflow: 'hidden' }}>
              <p style={{ fontSize: '13px', fontWeight: '600', color: '#fff', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden' }}>{user.name}</p>
              <p style={{ fontSize: '11px', color: '#888', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden' }}>{user.email}</p>
            </div>
          </div>
          <button onClick={triggerSignOut} className="btn-secondary" style={{ width: '100%', padding: '10px', fontSize: '13px' }}>
            <LogOut size={14} /> Wyloguj się
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main style={{ flex: 1, padding: '40px', overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
        <TabContent tab={activeTab} token={token} user={user} />
      </main>
    </div>
  );
}

function TabButton({ id, label, icon, activeTab, setActiveTab }) {
  const active = activeTab === id;
  return (
    <button 
      onClick={() => setActiveTab(id)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        width: '100%',
        padding: '12px 16px',
        border: 'none',
        borderRadius: '8px',
        background: active ? 'rgba(0, 255, 0, 0.1)' : 'transparent',
        color: active ? '#00ff00' : '#aaa',
        fontSize: '14px',
        fontWeight: active ? '600' : '400',
        cursor: 'pointer',
        textAlign: 'left',
        transition: 'all 0.2s',
      }}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

// --- Content Routing ---
function TabContent({ tab, token, user }) {
  switch (tab) {
    case 'dashboard':
      return <DashboardTab token={token} />;
    case 'sites':
      return <SitesTab token={token} user={user} />;
    case 'patients':
      return <PatientsTab token={token} user={user} />;
    case 'caching':
      return <CachingTab token={token} />;
    case 'cache_viewer':
      return <CacheViewerTab token={token} />;
    case 'unassigned':
      return <UnassignedTab token={token} user={user} />;
    case 'progress':
      return <ProgressTab token={token} />;
    case 'analysts':
      return <AnalystsTab token={token} />;
    case 'import':
      return <ImportTab token={token} />;
    default:
      return <div>Zakładka w budowie.</div>;
  }
}

// --- Tab 1: Dashboard ---
function DashboardTab({ token }) {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/metrics`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (!res.ok) throw new Error('Nie udało się pobrać statystyk');
      const data = await res.json();
      setMetrics(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, [token]);

  if (loading) return <TabLoader />;
  if (error) return <TabError error={error} retry={fetchMetrics} />;

  const pctProgress = metrics.total_assigned_patients > 0 
    ? (metrics.completed_assigned_count / metrics.total_assigned_patients) * 100 
    : 100;

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '28px', fontWeight: '700', letterSpacing: '-0.5px' }}>📈 Statystyki bazy danych</h2>
          <p style={{ color: '#888', marginTop: '4px' }}>Ogólne wskaźniki i podsumowanie analiz</p>
        </div>
        <button onClick={fetchMetrics} className="btn-secondary" style={{ padding: '8px 16px', fontSize: '13px' }}>
          <RefreshCw size={14} /> Odśwież dane
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
        <MetricCard title="Zakończone analizy w bazie" value={metrics.total_completed_reports} subtitle="Wszystkie rekordy w Firestore" />
        <MetricCard title="Unikalni zbadani pacjenci" value={metrics.unique_completed_patients} subtitle="🟢 Pacjenci ze statusem ukończenia" />
        <MetricCard 
          title="Postęp przypisanych zadań" 
          value={`${metrics.completed_assigned_count} / ${metrics.total_assigned_patients}`} 
          subtitle={`${pctProgress.toFixed(1)}% przypisanych przypadków`} 
        />
      </div>

      <h3 style={{ fontSize: '18px', fontWeight: '600', marginTop: '12px' }}>📁 Statystyki plików na VPS</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
        <MetricCard title="Wszyscy pacjenci na dysku" value={metrics.total_on_disk} subtitle="Wykryte foldery analityków" />
        <MetricCard title="Przeanalizowani na dysku" value={metrics.completed_on_disk} subtitle="🟢 Zakończone w systemie" />
        <MetricCard title="Pozostali do analizy" value={metrics.remaining_on_disk} subtitle="🟡 / ⚪ Oczekujący w kolejce" color="#f59e0b" />
      </div>
    </div>
  );
}

function MetricCard({ title, value, subtitle, color = '#00ff00', icon }) {
  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <span style={{ fontSize: '13px', color: '#888', fontWeight: '500', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{title}</span>
      <h3 style={{ fontSize: '36px', fontWeight: '700', color: color }}>{value}</h3>
      <span style={{ fontSize: '12px', color: '#666' }}>{subtitle}</span>
    </div>
  );
}

// --- Tab 2: Site Assignments ---
function SitesTab({ token, user }) {
  const [sites, setSites] = useState([]);
  const [analysts, setAnalysts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [resSites, resAnalysts] = await Promise.all([
        fetch(`${API_BASE}/sites`, { headers: { 'Authorization': `Bearer ${token}` } }),
        fetch(`${API_BASE}/analysts`, { headers: { 'Authorization': `Bearer ${token}` } })
      ]);
      if (!resSites.ok || !resAnalysts.ok) throw new Error('Błąd pobierania danych');
      const dataSites = await resSites.ok ? await resSites.json() : [];
      const dataAnalysts = await resAnalysts.ok ? await resAnalysts.json() : [];
      setSites(dataSites);
      setAnalysts(dataAnalysts);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [token]);

  const handleAssign = async (site, username) => {
    setError(null);
    setSuccess(null);
    try {
      const res = await fetch(`${API_BASE}/assign`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          type: 'site',
          target: site,
          assigned_to: username,
          assigned_by: user.username
        })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Błąd przypisywania');
      }
      setSuccess(`Ośrodek ${site} przypisany pomyślnie!`);
      fetchData();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleUnassign = async (site) => {
    setError(null);
    setSuccess(null);
    try {
      const res = await fetch(`${API_BASE}/unassign`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          type: 'site',
          target: site
        })
      });
      if (!res.ok) throw new Error('Błąd odpisania ośrodka');
      setSuccess(`Ośrodek ${site} został odpisany.`);
      fetchData();
    } catch (err) {
      setError(err.message);
    }
  };

  if (loading) return <TabLoader />;
  if (error) return <TabError error={error} retry={fetchData} />;

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h2 style={{ fontSize: '28px', fontWeight: '700', letterSpacing: '-0.5px' }}>🏢 Przypisywanie Ośrodków</h2>
        <p style={{ color: '#888', marginTop: '4px' }}>Przypisz cały ośrodek ze wszystkimi pacjentami wybranemu analitykowi</p>
      </div>

      {success && (
        <div style={{ display: 'flex', gap: '8px', padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(0,255,0,0.1)', border: '1px solid rgba(0,255,0,0.2)', color: '#00ff00', fontSize: '14px' }}>
          <CheckCircle2 size={18} />
          <span>{success}</span>
        </div>
      )}

      <div className="glass-card" style={{ padding: '0px', overflowX: 'auto', border: '1px solid var(--border-color)' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '14px' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-color)', color: '#888' }}>
              <th style={{ padding: '16px 24px' }}>Ośrodek</th>
              <th style={{ padding: '16px 24px' }}>Status przypisania</th>
              <th style={{ padding: '16px 24px' }}>Postęp analiz (Disk)</th>
              <th style={{ padding: '16px 24px', textAlign: 'right' }}>Akcja</th>
            </tr>
          </thead>
          <tbody>
            {sites.map(s => {
              const matchedSa = sa => sa.username === s.assigned_to;
              const assignedName = s.status === 'assigned_full' && s.assigned_to
                ? (analysts.find(matchedSa)?.name || s.assigned_to)
                : '';
              
              return (
                <tr key={s.site} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', verticalAlign: 'middle' }}>
                  <td style={{ padding: '16px 24px', fontWeight: '600' }}>🏢 Ośrodek {s.site}</td>
                  <td style={{ padding: '16px 24px' }}>
                    {s.status === 'assigned_full' ? (
                      <span style={{ color: '#00ff00', fontWeight: '500' }}>🟢 Przypisany w całości do `{assignedName}`</span>
                    ) : s.status === 'assigned_partial' ? (
                      <span style={{ color: '#f59e0b', fontWeight: '500' }}>🟡 Częściowo przypisany do {Array.isArray(s.assigned_to) ? s.assigned_to.join(', ') : s.assigned_to}</span>
                    ) : (
                      <span style={{ color: '#ef4444' }}>🔴 Nieprzypisany</span>
                    )}
                  </td>
                  <td style={{ padding: '16px 24px', color: '#aaa' }}>
                    {s.completed_patients} / {s.total_patients} ukończonych pacjentów
                  </td>
                  <td style={{ padding: '16px 24px', textAlign: 'right' }}>
                    {s.status === 'assigned_full' ? (
                      <button onClick={() => handleUnassign(s.site)} className="btn-danger" style={{ padding: '6px 12px', fontSize: '12px' }}>
                        Usuń przypisanie
                      </button>
                    ) : (
                      <select 
                        defaultValue=""
                        onChange={(e) => {
                          if (e.target.value) handleAssign(s.site, e.target.value);
                        }}
                        style={{ width: '160px', padding: '6px 12px', fontSize: '13px' }}
                      >
                        <option value="" disabled>Przypisz do...</option>
                        {analysts.map(a => (
                          <option key={a.uid} value={a.username}>{a.name}</option>
                        ))}
                      </select>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// --- Tab 3: Patient Assignments ---
function PatientsTab({ token, user }) {
  const [unassignedOptions, setUnassignedOptions] = useState([]);
  const [analysts, setAnalysts] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState('');
  const [selectedAnalyst, setSelectedAnalyst] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [resOptions, resAnalysts] = await Promise.all([
        fetch(`${API_BASE}/patients/unassigned-options`, { headers: { 'Authorization': `Bearer ${token}` } }),
        fetch(`${API_BASE}/analysts`, { headers: { 'Authorization': `Bearer ${token}` } })
      ]);
      if (!resOptions.ok || !resAnalysts.ok) throw new Error('Błąd pobierania danych');
      const dataOptions = await resOptions.json();
      const dataAnalysts = await resAnalysts.json();
      setUnassignedOptions(dataOptions);
      setAnalysts(dataAnalysts);
      if (dataOptions.length > 0) setSelectedPatient(dataOptions[0].patient_id);
      if (dataAnalysts.length > 0) setSelectedAnalyst(dataAnalysts[0].username);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [token]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);
    if (!selectedPatient || !selectedAnalyst) return;

    try {
      const res = await fetch(`${API_BASE}/assign`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          type: 'patient',
          target: selectedPatient,
          assigned_to: selectedAnalyst,
          assigned_by: user.username
        })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Błąd przypisywania');
      }
      setSuccess(`Pacjent ${selectedPatient} pomyślnie przypisany do ${selectedAnalyst}!`);
      fetchData();
    } catch (err) {
      setError(err.message);
    }
  };

  if (loading) return <TabLoader />;
  if (error) return <TabError error={error} retry={fetchData} />;

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px', maxWidth: '600px' }}>
      <div>
        <h2 style={{ fontSize: '28px', fontWeight: '700', letterSpacing: '-0.5px' }}>📁 Przypisywanie Pojedynczych Pacjentów</h2>
        <p style={{ color: '#888', marginTop: '4px' }}>Przypisz wolnego (nieprzypisanego) pacjenta wybranemu analitykowi</p>
      </div>

      {success && (
        <div style={{ display: 'flex', gap: '8px', padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(0,255,0,0.1)', border: '1px solid rgba(0,255,0,0.2)', color: '#00ff00', fontSize: '14px' }}>
          <CheckCircle2 size={18} />
          <span>{success}</span>
        </div>
      )}

      {unassignedOptions.length === 0 ? (
        <div className="glass-card" style={{ borderLeft: '4px solid #aaa' }}>
          <p style={{ color: '#aaa', fontSize: '15px' }}>ℹ️ Brak wolnych pacjentów do przypisania. Wszyscy są przypisani lub ośrodki są przydzielone w całości.</p>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div>
            <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: '#aaa' }}>Wybierz wolnego pacjenta</label>
            <select value={selectedPatient} onChange={e => setSelectedPatient(e.target.value)}>
              {unassignedOptions.map(p => (
                <option key={p.patient_id} value={p.patient_id}>
                  {p.completed ? '🟢 ' : ''}{p.patient_id} (Ośrodek {p.site}){p.completed ? ' (ukończony)' : ''}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: '#aaa' }}>Wybierz analityka</label>
            <select value={selectedAnalyst} onChange={e => setSelectedAnalyst(e.target.value)}>
              {analysts.map(a => (
                <option key={a.uid} value={a.username}>{a.name} ({a.email})</option>
              ))}
            </select>
          </div>

          <button type="submit" className="btn-primary" style={{ marginTop: '8px' }}>
            <UserCheck2 size={18} /> Przypisz pacjenta
          </button>
        </form>
      )}
    </div>
  );
}

// --- Tab 4: Cache Manager ---
function CachingTab({ token }) {
  const [scope, setScope] = useState('site'); // 'site' or 'patient'
  const [sites, setSites] = useState([]);
  const [analysts, setAnalysts] = useState([]);
  const [selectedTarget, setSelectedTarget] = useState('');
  
  // Estimation states
  const [estimating, setEstimating] = useState(false);
  const [estimateData, setEstimateData] = useState(null);
  
  const [confirmAccept, setConfirmAccept] = useState(false);
  
  // Active copy tasks states
  const [activeTasks, setActiveTasks] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [startingCache, setStartingCache] = useState(false);

  const fetchInitialData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [resSites, resAnalysts] = await Promise.all([
        fetch(`${API_BASE}/sites`, { headers: { 'Authorization': `Bearer ${token}` } }),
        fetch(`${API_BASE}/analysts/progress`, { headers: { 'Authorization': `Bearer ${token}` } })
      ]);
      if (!resSites.ok || !resAnalysts.ok) throw new Error('Błąd ładowania danych');
      const dataSites = await resSites.json();
      const dataAnalysts = await resAnalysts.json();
      
      setSites(dataSites);
      setAnalysts(dataAnalysts);
      
      // Auto-select first option
      if (scope === 'site') {
        if (dataSites.length > 0) setSelectedTarget(dataSites[0].site);
      } else {
        const assignedPatients = [];
        dataAnalysts.forEach(a => {
          a.tasks.forEach(t => {
            if (t.type === 'patient' && !t.completed) {
              assignedPatients.push(t.patient_id);
            }
          });
        });
        const uniquePats = Array.from(new Set(assignedPatients));
        if (uniquePats.length > 0) setSelectedTarget(uniquePats[0]);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchTasks = async () => {
    try {
      const res = await fetch(`${API_BASE}/cache/tasks`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setActiveTasks(data);
      }
    } catch (err) {
      console.error('Error fetching copy tasks:', err);
    }
  };

  // Poll tasks every 2.5 seconds
  useEffect(() => {
    fetchInitialData();
    fetchTasks();
    const interval = setInterval(fetchTasks, 2500);
    return () => clearInterval(interval);
  }, [token, scope]);

  const handleEstimate = async () => {
    if (!selectedTarget) return;
    setEstimating(true);
    setEstimateData(null);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/cache/estimate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ type: scope, target: selectedTarget })
      });
      if (!res.ok) throw new Error('Błąd szacowania wielkości');
      const data = await res.json();
      setEstimateData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setEstimating(false);
    }
  };

  // Re-estimate on target selection changes
  useEffect(() => {
    setEstimateData(null);
    setConfirmAccept(false);
  }, [selectedTarget, scope]);

  const handleStartCache = async () => {
    if (!selectedTarget || !confirmAccept || startingCache) return;
    setError(null);
    setSuccess(null);
    setStartingCache(true);
    try {
      const res = await fetch(`${API_BASE}/cache/prefetch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ type: scope, target: selectedTarget })
      });
      if (!res.ok) throw new Error('Błąd uruchamiania pobierania');
      const data = await res.json();
      if (data.status === 'ignored') {
        setSuccess('Wszystkie pliki tego przypadku są już zapisane na dysku.');
      } else {
        setSuccess(`Uruchomiono pobieranie w tle (Task ID: ${data.task_id})`);
      }
      setEstimateData(null);
      setConfirmAccept(false);
      fetchTasks();
    } catch (err) {
      setError(err.message);
    } finally {
      setStartingCache(false);
    }
  };

  const handleDismissTask = async (tid) => {
    try {
      const res = await fetch(`${API_BASE}/cache/tasks/${tid}/dismiss`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) fetchTasks();
    } catch (err) {
      console.error(err);
    }
  };

  if (loading) return <TabLoader />;
  if (error) return <TabError error={error} retry={fetchInitialData} />;

  // Filter site list and patient list based on assignments
  const assignedSites = sites.filter(s => s.status === 'assigned_full' || s.status === 'assigned_partial');
  
  const assignedPatients = [];
  analysts.forEach(a => {
    a.tasks.forEach(t => {
      if (t.type === 'patient' && !t.completed) {
        assignedPatients.push(t.patient_id);
      }
    });
  });
  const uniquePats = Array.from(new Set(assignedPatients));

  // Check if there are active prefetch tasks running
  const runningTaskKeys = Object.keys(activeTasks);

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h2 style={{ fontSize: '28px', fontWeight: '700', letterSpacing: '-0.5px' }}>📥 Pobieranie Plików DICOM (Cache Manager)</h2>
        <p style={{ color: '#888', marginTop: '4px' }}>Zapisz pliki z dysku sieciowego na szybki dysk VPS, aby zapobiec opóźnieniom u analityków</p>
      </div>

      {success && (
        <div style={{ display: 'flex', gap: '8px', padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(0,255,0,0.1)', border: '1px solid rgba(0,255,0,0.2)', color: '#00ff00', fontSize: '14px' }}>
          <CheckCircle2 size={18} />
          <span>{success}</span>
        </div>
      )}

      {/* Render active tasks */}
      {runningTaskKeys.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <h3 style={{ fontSize: '16px', fontWeight: '600' }}>🚀 Aktywne zadania zapisu w tle:</h3>
          {runningTaskKeys.map(tid => {
            const task = activeTasks[tid];
            const isScanning = task.detail === 'Scanning files on Tailscale...';
            const progressVal = task.total_bytes > 0 ? (task.copied_bytes / task.total_bytes) : 0;
            const progressPercent = (progressVal * 100).toFixed(1);
            
            return (
              <div key={tid} className="glass-card fade-in" style={{ borderLeft: task.status === 'success' ? '4px solid #00ff00' : task.status === 'error' ? '4px solid #ef4444' : '4px solid #f59e0b' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                  <div>
                    <h4 style={{ fontWeight: '600', color: '#fff', fontSize: '14px' }}>
                      {tid.includes('site') ? `Pobieranie Ośrodka: ${tid.split('_')[3]}` : `Pobieranie Pacjenta: ${tid.split('_')[3]}`}
                    </h4>
                    <p style={{ fontSize: '11px', color: '#888', marginTop: '2px' }}>Task ID: {tid}</p>
                  </div>
                  {(task.status === 'success' || task.status === 'error') && (
                    <button onClick={() => handleDismissTask(tid)} className="btn-secondary" style={{ padding: '4px 10px', fontSize: '11px' }}>
                      Odrzuć powiadomienie
                    </button>
                  )}
                </div>

                {task.status === 'running' ? (
                  <div>
                    {isScanning ? (
                      <p style={{ fontSize: '13px', color: '#f59e0b', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <Loader2 className="spin" size={16} /> 🔍 Skanowanie plików na dysku sieciowym... (to może potrwać do 30 sekund)
                      </p>
                    ) : (
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#aaa', marginBottom: '4px' }}>
                          <span>Pobrano: {task.copied_files} / {task.total_files} plików ({progressPercent}%)</span>
                          <span>Prędkość: {task.speed.toFixed(1)} MB/s</span>
                        </div>
                        <div style={{ width: '100%', height: '8px', backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden', marginBottom: '4px' }}>
                          <div style={{ width: `${progressPercent}%`, height: '100%', backgroundColor: '#00ff00', borderRadius: '4px', boxShadow: '0 0 8px #00ff00' }}></div>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#666' }}>
                          <span>Wielkość: {(task.copied_bytes / (1024*1024)).toFixed(1)} MB / {(task.total_bytes / (1024*1024)).toFixed(1)} MB</span>
                          <span>Pozostały czas: {task.est_left > 0 ? `${(task.est_left / 60).toFixed(1)} min` : 'wyliczanie...'}</span>
                        </div>
                      </div>
                    )}
                  </div>
                ) : task.status === 'success' ? (
                  <p style={{ color: '#00ff00', fontSize: '13px', fontWeight: '500' }}>🟢 Zakończono pomyślnie! Pliki są zapisane w pamięci VPS.</p>
                ) : (
                  <p style={{ color: '#ef4444', fontSize: '13px', fontWeight: '500' }}>❌ Błąd pobierania: {task.error_msg || 'Przerwane.'}</p>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Trigger form */}
      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div>
          <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '12px' }}>Uruchom nowe pobieranie:</h3>
          <div style={{ display: 'flex', gap: '24px', marginBottom: '16px' }}>
            <label className="checkbox-container">
              <input type="radio" checked={scope === 'site'} onChange={() => { setScope('site'); setSelectedTarget(''); }} />
              <span className="checkmark" style={{ borderRadius: '50%' }}></span>
              <span>Scope: Cały Ośrodek</span>
            </label>
            <label className="checkbox-container">
              <input type="radio" checked={scope === 'patient'} onChange={() => { setScope('patient'); setSelectedTarget(''); }} />
              <span className="checkmark" style={{ borderRadius: '50%' }}></span>
              <span>Scope: Jeden Pacjent</span>
            </label>
          </div>
        </div>

        {scope === 'site' ? (
          <div>
            <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: '#aaa' }}>Wybierz ośrodek (z serwera Tailscale)</label>
            {sites.length === 0 ? (
              <p style={{ color: '#888', fontSize: '13px' }}>Brak ośrodków na serwerze Tailscale.</p>
            ) : (
              <select value={selectedTarget} onChange={e => setSelectedTarget(e.target.value)} style={{ maxWidth: '400px' }}>
                <option value="" disabled>-- Wybierz Ośrodek --</option>
                {sites.map(s => (
                  <option key={s.site} value={s.site}>
                    Ośrodek {s.site} {s.status === 'unassigned' ? '(Nieprzypisany)' : ''}
                  </option>
                ))}
              </select>
            )}
          </div>
        ) : (
          <div>
            <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: '#aaa' }}>Wybierz pacjenta z przypisanych</label>
            {uniquePats.length === 0 ? (
              <p style={{ color: '#888', fontSize: '13px' }}>Brak nieukończonych pacjentów do pobrania.</p>
            ) : (
              <select value={selectedTarget} onChange={e => setSelectedTarget(e.target.value)} style={{ maxWidth: '400px' }}>
                <option value="" disabled>-- Wybierz Pacjenta --</option>
                {uniquePats.map(pid => (
                  <option key={pid} value={pid}>Pacjent {pid}</option>
                ))}
              </select>
            )}
          </div>
        )}

        {selectedTarget && !estimateData && (
          <button onClick={handleEstimate} className="btn-secondary" disabled={estimating} style={{ maxWidth: '240px' }}>
            {estimating ? <Loader2 className="spin" size={16} /> : 'Szacuj wielkość plików'}
          </button>
        )}

        {estimateData && (
          <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '16px', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '16px' }}>
            <div style={{ padding: '16px', borderRadius: '12px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
              <h4 style={{ fontWeight: '600', color: '#fff', fontSize: '14px', marginBottom: '12px' }}>Podsumowanie statusu pacjentów:</h4>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '12px', marginBottom: '16px' }}>
                <div style={{ padding: '10px', borderRadius: '8px', backgroundColor: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.03)' }}>
                  <div style={{ fontSize: '11px', color: '#888' }}>Wszystkich spraw</div>
                  <div style={{ fontSize: '18px', fontWeight: '700', color: '#fff', marginTop: '2px' }}>{estimateData.total_patients}</div>
                </div>
                <div style={{ padding: '10px', borderRadius: '8px', backgroundColor: 'rgba(0, 255, 0, 0.02)', border: '1px solid rgba(0,255,0,0.05)' }}>
                  <div style={{ fontSize: '11px', color: '#888' }}>W cache VPS 💾</div>
                  <div style={{ fontSize: '18px', fontWeight: '700', color: '#00ff00', marginTop: '2px' }}>{estimateData.cached_patients?.length || 0}</div>
                </div>
                <div style={{ padding: '10px', borderRadius: '8px', backgroundColor: 'rgba(59, 130, 246, 0.02)', border: '1px solid rgba(59,130,246,0.05)' }}>
                  <div style={{ fontSize: '11px', color: '#888' }}>Zakończone 🟢</div>
                  <div style={{ fontSize: '18px', fontWeight: '700', color: '#3b82f6', marginTop: '2px' }}>{estimateData.completed_patients?.length || 0}</div>
                </div>
                <div style={{ padding: '10px', borderRadius: '8px', backgroundColor: 'rgba(245, 158, 11, 0.02)', border: '1px solid rgba(245,158,11,0.05)' }}>
                  <div style={{ fontSize: '11px', color: '#888' }}>Do pobrania 📥</div>
                  <div style={{ fontSize: '18px', fontWeight: '700', color: '#f59e0b', marginTop: '2px' }}>{estimateData.scheduled_patients?.length || 0}</div>
                </div>
              </div>

              {estimateData.scheduled_patients.length > 0 ? (
                <div style={{ padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(0, 255, 0, 0.04)', border: '1px solid rgba(0,255,0,0.1)' }}>
                  <p style={{ fontSize: '14px', color: '#fff' }}>
                    📦 Szacowana wielkość plików do wgrania: <strong style={{ color: '#00ff00' }}>{estimateData.size_gb.toFixed(2)} GB</strong>
                  </p>
                  <div style={{ marginTop: '8px' }}>
                    <span style={{ fontSize: '12px', color: '#aaa' }}>Pacjenci zakwalifikowani do pobrania ({estimateData.scheduled_patients.length}):</span>
                    <p style={{ fontSize: '13px', color: '#fff', fontWeight: '500', marginTop: '2px' }}>{estimateData.scheduled_patients.join(', ')}</p>
                  </div>
                </div>
              ) : (
                <div style={{ padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(0, 255, 0, 0.04)', border: '1px solid rgba(0,255,0,0.1)' }}>
                  <p style={{ fontSize: '13px', color: '#00ff00', fontWeight: '500' }}>
                    Wszystkie aktywne (nieukończone) przypadki tego ośrodka są już zapisane w pamięci podręcznej VPS.
                  </p>
                  {estimateData.completed_patients?.length > 0 && (
                    <p style={{ fontSize: '12px', color: '#888', marginTop: '4px' }}>
                      Pozostałe {estimateData.completed_patients.length} przypadków zostało już ukończone (Completed) i nie wymagają pobierania.
                    </p>
                  )}
                </div>
              )}
            </div>

            {estimateData.scheduled_patients.length > 0 && (
              <div>
                <label className="checkbox-container" style={{ marginBottom: '16px' }}>
                  <input type="checkbox" checked={confirmAccept} onChange={e => setConfirmAccept(e.target.checked)} />
                  <span className="checkmark"></span>
                  <span>Potwierdzam chęć wgrania powyższych aktywnych przypadków do pamięci VPS</span>
                </label>

                <button onClick={handleStartCache} className="btn-primary" disabled={!confirmAccept || startingCache} style={{ maxWidth: '340px' }}>
                  {startingCache ? <Loader2 className="spin" size={18} /> : <FolderSync size={18} />} Uruchom pobieranie (ANGIO folders only)
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// --- Tab 5: Unassigned Cases ---
function UnassignedTab({ token, user }) {
  const [unassigned, setUnassigned] = useState([]);
  const [analysts, setAnalysts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [resUnassigned, resAnalysts] = await Promise.all([
        fetch(`${API_BASE}/unassigned`, { headers: { 'Authorization': `Bearer ${token}` } }),
        fetch(`${API_BASE}/analysts`, { headers: { 'Authorization': `Bearer ${token}` } })
      ]);
      if (!resUnassigned.ok || !resAnalysts.ok) throw new Error('Błąd ładowania danych');
      const dataUnassigned = await resUnassigned.json();
      const dataAnalysts = await resAnalysts.json();
      setUnassigned(dataUnassigned);
      setAnalysts(dataAnalysts);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [token]);

  const handleReassign = async (pid, site, username) => {
    setError(null);
    setSuccess(null);
    try {
      const res = await fetch(`${API_BASE}/reassign`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          patient_id: pid,
          site: site,
          assigned_to: username,
          assigned_by: user.username
        })
      });
      if (!res.ok) throw new Error('Błąd ponownego przypisywania');
      setSuccess(`Zadanie ${pid} przypisane pomyślnie do ${username}!`);
      fetchData();
    } catch (err) {
      setError(err.message);
    }
  };

  if (loading) return <TabLoader />;
  if (error) return <TabError error={error} retry={fetchData} />;

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h2 style={{ fontSize: '28px', fontWeight: '700', letterSpacing: '-0.5px' }}>🚨 Odpisane Przypadki (Unassigned Cases)</h2>
        <p style={{ color: '#888', marginTop: '4px' }}>Przeglądaj zadania odpisane przez analityków z podaniem przyczyny i przypisz je ponownie</p>
      </div>

      {success && (
        <div style={{ display: 'flex', gap: '8px', padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(0,255,0,0.1)', border: '1px solid rgba(0,255,0,0.2)', color: '#00ff00', fontSize: '14px' }}>
          <CheckCircle2 size={18} />
          <span>{success}</span>
        </div>
      )}

      {unassigned.length === 0 ? (
        <div className="glass-card" style={{ borderLeft: '4px solid #00ff00' }}>
          <p style={{ color: '#aaa', fontSize: '15px' }}>🟢 Brak odpisanych przypadków. Wszystkie zadania są aktywne.</p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {unassigned.map(u => (
            <div key={u.patient_id} className="glass-card fade-in" style={{ display: 'flex', flexWrap: 'wrap', gap: '24px', alignItems: 'center', justifyContent: 'space-between', borderLeft: '4px solid #ef4444' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', flex: 1, minWidth: '240px' }}>
                <h4 style={{ fontWeight: '700', fontSize: '16px', color: '#fff' }}>📁 Pacjent {u.patient_id} (Ośrodek {u.site})</h4>
                <p style={{ fontSize: '13px', color: '#888' }}>
                  Odpisał: <strong style={{ color: '#aaa' }}>{u.unassigned_by}</strong> | 🕒 {u.unassigned_at ? u.unassigned_at.split('.')[0] : 'N/A'}
                </p>
                <p style={{ fontSize: '14px', color: '#f59e0b', marginTop: '4px', fontStyle: 'italic' }}>
                  💬 Powód: "{u.unassigned_reason}"
                </p>
              </div>

              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <select 
                  defaultValue="" 
                  onChange={e => {
                    if (e.target.value) handleReassign(u.patient_id, u.site, e.target.value);
                  }}
                  style={{ width: '180px', padding: '8px 12px' }}
                >
                  <option value="" disabled>Przypisz do...</option>
                  {analysts.map(a => (
                    <option key={a.uid} value={a.username}>{a.name}</option>
                  ))}
                </select>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// --- Tab 6: Analysts Progress ---
function ProgressTab({ token }) {
  const [analysts, setAnalysts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedAnalyst, setExpandedAnalyst] = useState(null);

  const fetchProgress = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/analysts/progress`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (!res.ok) throw new Error('Błąd ładowania postępów');
      const data = await res.json();
      setAnalysts(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProgress();
  }, [token]);

  if (loading) return <TabLoader />;
  if (error) return <TabError error={error} retry={fetchProgress} />;

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h2 style={{ fontSize: '28px', fontWeight: '700', letterSpacing: '-0.5px' }}>📊 Postęp Prac Analityków</h2>
        <p style={{ color: '#888', marginTop: '4px' }}>Przeglądaj statystyki wykonanych analiz i listę przypisanych zadań każdego analityka</p>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {analysts.map(a => {
          const pct = a.total_assigned > 0 ? (a.completed_assigned / a.total_assigned) * 100 : 100;
          const isExpanded = expandedAnalyst === a.username;
          
          return (
            <div key={a.username} className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '16px' }}>
                <div>
                  <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#fff' }}>👤 {a.name}</h3>
                  <p style={{ fontSize: '13px', color: '#888' }}>{a.email}</p>
                </div>
                <div style={{ textAlign: 'right', fontSize: '13px' }}>
                  <p style={{ color: '#aaa' }}>Przypisane: <strong>{a.completed_assigned} / {a.total_assigned}</strong> zbadane (pozostało: {a.remaining_assigned})</p>
                  <p style={{ color: '#666', marginTop: '2px' }}>Razem ukończone: {a.unique_patients} pacjentów (zapisane raporty: {a.total_reports})</p>
                </div>
              </div>

              {/* Progress bar */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#aaa', marginBottom: '6px' }}>
                  <span>Postęp przypisanych zadań</span>
                  <span>{pct.toFixed(1)}%</span>
                </div>
                <div style={{ width: '100%', height: '8px', backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ width: `${pct}%`, height: '100%', backgroundColor: '#00ff00', borderRadius: '4px', boxShadow: '0 0 6px #00ff00' }}></div>
                </div>
              </div>

              {/* Expand task details */}
              <div>
                <button 
                  onClick={() => setExpandedAnalyst(isExpanded ? null : a.username)}
                  className="btn-secondary"
                  style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px', fontSize: '12px' }}
                >
                  <ChevronRight size={14} style={{ transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'all 0.2s' }} />
                  {isExpanded ? 'Ukryj zadania' : `Pokaż zadania (${a.tasks.length})`}
                </button>

                {isExpanded && (
                  <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '16px', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '16px' }}>
                    {a.tasks.length === 0 ? (
                      <p style={{ color: '#666', fontSize: '13px', paddingLeft: '8px' }}>Brak zadań.</p>
                    ) : (
                      a.tasks.map((t, idx) => (
                        <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 12px', borderRadius: '8px', backgroundColor: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.03)' }}>
                          {t.type === 'site' ? (
                            <>
                              <span style={{ fontSize: '13.5px', fontWeight: '600' }}>🏢 Ośrodek {t.site} (w całości)</span>
                              <span style={{ fontSize: '12.5px', color: '#aaa' }}>Postęp: {t.completed_patients} / {t.total_patients} pacjentów</span>
                            </>
                          ) : (
                            <>
                              <span style={{ fontSize: '13.5px', fontWeight: '600' }}>📁 Pacjent {t.patient_id} (Ośrodek {t.site})</span>
                              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                                <span style={{ fontSize: '12.5px', color: t.completed ? '#00ff00' : '#f59e0b' }}>
                                  {t.completed ? '🟢 Ukończony' : '🟡 W trakcie'}
                                </span>
                                {t.completed && t.report_url && (
                                  <a href={t.report_url} target="_blank" rel="noreferrer" style={{ color: '#00ff00', fontSize: '12px', textDecoration: 'underline' }}>
                                    Pokaż Raport PDF
                                  </a>
                                )}
                              </div>
                            </>
                          )}
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// --- Tab 7: Add Analyst ---
function AnalystsTab({ token }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/analysts`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ email, password, name })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Błąd rejestracji analityka');
      }
      setSuccess(`Dodano pomyślnie konto dla ${email}!`);
      setEmail('');
      setPassword('');
      setName('');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px', maxWidth: '500px' }}>
      <div>
        <h2 style={{ fontSize: '28px', fontWeight: '700', letterSpacing: '-0.5px' }}>👤 Dodawanie Nowego Analityka</h2>
        <p style={{ color: '#888', marginTop: '4px' }}>Zarejestruj nowego lekarza/analityka w systemie Firebase</p>
      </div>

      {success && (
        <div style={{ display: 'flex', gap: '8px', padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(0,255,0,0.1)', border: '1px solid rgba(0,255,0,0.2)', color: '#00ff00', fontSize: '14px' }}>
          <CheckCircle2 size={18} />
          <span>{success}</span>
        </div>
      )}

      {error && (
        <div style={{ display: 'flex', gap: '8px', padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)', color: '#ef4444', fontSize: '14px' }}>
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div>
          <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: '#aaa' }}>Imię i nazwisko</label>
          <input type="text" value={name} onChange={e => setName(e.target.value)} required placeholder="np. Jan Kowalski" />
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: '#aaa' }}>Email analityka</label>
          <input type="email" value={email} onChange={e => setEmail(e.target.value)} required placeholder="np. j.kowalski@angiopy.tech" />
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: '#aaa' }}>Tymczasowe hasło (min. 6 znaków)</label>
          <input type="password" value={password} onChange={e => setPassword(e.target.value)} required minLength={6} placeholder="••••••••" />
        </div>

        <button type="submit" className="btn-primary" disabled={loading} style={{ marginTop: '8px' }}>
          {loading ? <Loader2 className="spin" size={18} /> : 'Zarejestruj użytkownika'}
        </button>
      </form>
    </div>
  );
}

// --- Tab 8: Bulk PDF Import ---
function ImportTab({ token }) {
  const [analysts, setAnalysts] = useState([]);
  const [selectedAnalyst, setSelectedAnalyst] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [importing, setImporting] = useState(false);
  const [importLogs, setImportLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  const fetchAnalystsList = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/analysts`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (!res.ok) throw new Error('Błąd ładowania analityków');
      const data = await res.json();
      setAnalysts(data);
      if (data.length > 0) setSelectedAnalyst(data[0].username);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalystsList();
  }, [token]);

  const handleFileChange = (e) => {
    setSelectedFiles(Array.from(e.target.files));
    setImportLogs([]);
    setSuccessMsg(null);
  };

  const handleStartImport = async (e) => {
    e.preventDefault();
    if (selectedFiles.length === 0 || !selectedAnalyst) return;

    setImporting(true);
    setImportLogs([]);
    setSuccessMsg(null);

    const formData = new FormData();
    formData.append('analyst', selectedAnalyst);
    selectedFiles.forEach(file => {
      formData.append('files', file);
    });

    try {
      const res = await fetch(`${API_BASE}/import`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData
      });
      if (!res.ok) throw new Error('Błąd połączenia z serwerem podczas importu');
      const data = await res.json();
      setSuccessMsg(`Zakończono import! Zapisano raportów: ${data.success_count}, Błędy: ${data.error_count}.`);
      setImportLogs(data.results);
      setSelectedFiles([]);
    } catch (err) {
      setError(err.message);
    } finally {
      setImporting(false);
    }
  };

  if (loading) return <TabLoader />;
  if (error) return <TabError error={error} retry={fetchAnalystsList} />;

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px', maxWidth: '700px' }}>
      <div>
        <h2 style={{ fontSize: '28px', fontWeight: '700', letterSpacing: '-0.5px' }}>🚀 Masowy Import Poprzednich Analiz (PDF)</h2>
        <p style={{ color: '#888', marginTop: '4px' }}>Wgraj historyczne QCA raporty PDF i automatycznie sparsuj parametry do bazy Firestore</p>
      </div>

      {successMsg && (
        <div style={{ display: 'flex', gap: '8px', padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(0,255,0,0.1)', border: '1px solid rgba(0,255,0,0.2)', color: '#00ff00', fontSize: '14px' }}>
          <CheckCircle2 size={18} style={{ flexShrink: 0 }} />
          <span>{successMsg}</span>
        </div>
      )}

      <form onSubmit={handleStartImport} className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div>
          <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: '#aaa' }}>Przypisz wgrywane raporty analitykowi:</label>
          <select value={selectedAnalyst} onChange={e => setSelectedAnalyst(e.target.value)}>
            {analysts.map(a => (
              <option key={a.uid} value={a.username}>{a.name} ({a.email})</option>
            ))}
          </select>
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: '#aaa' }}>Wybierz raporty PDF z dysku (wielokrotny wybór)</label>
          <input 
            type="file" 
            accept=".pdf" 
            multiple 
            onChange={handleFileChange}
            style={{ 
              padding: '16px', 
              border: '2px dashed var(--glass-border)', 
              borderRadius: '8px', 
              background: 'rgba(255,255,255,0.01)', 
              cursor: 'pointer',
              color: '#888'
            }}
          />
        </div>

        {selectedFiles.length > 0 && (
          <p style={{ fontSize: '13px', color: '#aaa' }}>
            Wybrano plików: <strong>{selectedFiles.length}</strong>
          </p>
        )}

        <button type="submit" className="btn-primary" disabled={importing || selectedFiles.length === 0} style={{ marginTop: '8px' }}>
          {importing ? <Loader2 className="spin" size={18} /> : 'Uruchom masowy import'}
        </button>
      </form>

      {importLogs.length > 0 && (
        <div className="glass-card fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <h3 style={{ fontSize: '15px', fontWeight: '600' }}>Logi importu:</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '200px', overflowY: 'auto' }}>
            {importLogs.map((log, idx) => (
              <div key={idx} style={{ fontSize: '13px', display: 'flex', justifyContent: 'space-between', color: log.status === 'success' ? '#00ff00' : '#ef4444' }}>
                <span>• {log.filename}</span>
                <span>{log.status === 'success' ? '🟢 OK' : `❌ ${log.error || 'Błąd'}`}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// --- Inner Shared Components ---
function TabLoader() {
  return (
    <div style={{ display: 'flex', flex: 1, height: '400px', justifyContent: 'center', alignItems: 'center' }}>
      <Loader2 className="spin" size={32} color="#00ff00" />
    </div>
  );
}

function TabError({ error, retry }) {
  return (
    <div className="glass-card" style={{ borderLeft: '4px solid #ef4444', display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <p style={{ color: '#ef4444', fontWeight: '600' }}>Wystąpił błąd podczas pobierania danych:</p>
      <p style={{ color: '#aaa', fontSize: '14px' }}>{error}</p>
      {retry && (
        <button onClick={retry} className="btn-secondary" style={{ maxWidth: '160px', padding: '8px 16px', fontSize: '12px' }}>
          Spróbuj ponownie
        </button>
      )}
    </div>
  );
}

// --- Tab 9: Cache Viewer ---
function CacheViewerTab({ token }) {
  const [cacheData, setCacheData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedSite, setSelectedSite] = useState('all');
  const [deletingId, setDeletingId] = useState(null);
  const [confirmDelete, setConfirmDelete] = useState(null);

  const fetchCacheData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/cache/files`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (!res.ok) throw new Error('Nie udało się pobrać informacji o cache');
      const data = await res.json();
      setCacheData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCacheData();
  }, [token]);

  const handleDeleteCache = async (site, patientId) => {
    setDeletingId(patientId);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/cache/files/${site}/${patientId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Błąd podczas usuwania cache');
      }
      setConfirmDelete(null);
      await fetchCacheData();
    } catch (err) {
      setError(err.message);
    } finally {
      setDeletingId(null);
    }
  };

  const formatDate = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp * 1000);
    return date.toLocaleString('pl-PL', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (loading && !cacheData) return <TabLoader />;
  if (error && !cacheData) return <TabError error={error} retry={fetchCacheData} />;

  const uniqueSites = cacheData?.patients 
    ? Array.from(new Set(cacheData.patients.map(p => p.site))).sort()
    : [];

  const filteredPatients = cacheData?.patients?.filter(p => 
    selectedSite === 'all' || p.site === selectedSite
  ) || [];

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h2 style={{ fontSize: '28px', fontWeight: '700', letterSpacing: '-0.5px' }}>💾 Stan Cache VPS (Cached Patients)</h2>
        <p style={{ color: '#888', marginTop: '4px' }}>Przeglądaj pacjentów aktualnie zapisanych w pamięci podręcznej serwera VPS oraz zarządzaj miejscem na dysku</p>
      </div>

      {error && (
        <div style={{ display: 'flex', gap: '8px', padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)', color: '#ef4444', fontSize: '14px' }}>
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px' }}>
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <span style={{ fontSize: '14px', color: '#888', fontWeight: '500' }}>Całkowity Rozmiar Cache</span>
          <span style={{ fontSize: '32px', fontWeight: '700', color: '#00ff00' }}>
            {cacheData ? `${cacheData.total_size_mb >= 1024 ? (cacheData.total_size_mb / 1024).toFixed(2) + ' GB' : cacheData.total_size_mb + ' MB'}` : '0 MB'}
          </span>
        </div>
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <span style={{ fontSize: '14px', color: '#888', fontWeight: '500' }}>Zapisani Pacjenci</span>
          <span style={{ fontSize: '32px', fontWeight: '700', color: '#fff' }}>
            {cacheData?.total_patients || 0}
          </span>
        </div>
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <span style={{ fontSize: '14px', color: '#888', fontWeight: '500' }}>Lokalizacja na VPS</span>
          <span style={{ fontSize: '14px', fontWeight: '600', color: '#aaa', wordBreak: 'break-all', marginTop: 'auto' }}>
            ./tailscale_cache
          </span>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '16px', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1, minWidth: '240px', maxWidth: '400px' }}>
          <span style={{ fontSize: '14px', color: '#aaa', fontWeight: '500', whiteSpace: 'nowrap' }}>Filtruj według ośrodka:</span>
          <select 
            value={selectedSite}
            onChange={e => setSelectedSite(e.target.value)}
            style={{ 
              flex: 1, 
              padding: '10px 12px', 
              borderRadius: '8px', 
              border: '1px solid var(--border-color)', 
              backgroundColor: 'rgba(255,255,255,0.02)', 
              color: '#fff',
              outline: 'none',
              cursor: 'pointer'
            }}
          >
            <option value="all">Wszystkie ośrodki</option>
            {uniqueSites.map(site => (
              <option key={site} value={site}>Ośrodek {site}</option>
            ))}
          </select>
        </div>
        <button onClick={fetchCacheData} className="btn-secondary" style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 16px' }}>
          <RefreshCw size={16} />
          Odśwież
        </button>
      </div>

      {confirmDelete && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.8)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999, padding: '20px' }}>
          <div className="glass-card fade-in" style={{ maxWidth: '480px', width: '100%', display: 'flex', flexDirection: 'column', gap: '20px', border: '1px solid rgba(239,68,68,0.2)' }}>
            <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '50%', backgroundColor: 'rgba(239,68,68,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#ef4444', flexShrink: 0 }}>
                <AlertTriangle size={24} />
              </div>
              <div>
                <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#fff' }}>Potwierdź usunięcie z cache</h3>
                <p style={{ color: '#aaa', fontSize: '14px', marginTop: '6px', lineHeight: '1.5' }}>
                  Czy na pewno chcesz usunąć pliki z cache dla pacjenta <strong style={{ color: '#fff' }}>{confirmDelete.patient_id}</strong> (Ośrodek {confirmDelete.site})?
                  Pliki te zostaną usunięte z lokalnego dysku serwera VPS. Będą musiały zostać pobrane ponownie z Tailscale przed rozpoczęciem analizy.
                </p>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '8px' }}>
              <button 
                onClick={() => setConfirmDelete(null)} 
                className="btn-secondary" 
                disabled={deletingId !== null}
                style={{ padding: '8px 16px' }}
              >
                Anuluj
              </button>
              <button 
                onClick={() => handleDeleteCache(confirmDelete.site, confirmDelete.patient_id)} 
                disabled={deletingId !== null}
                style={{ backgroundColor: '#ef4444', color: '#fff', border: 'none', borderRadius: '6px', padding: '8px 16px', fontWeight: '600', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' }}
              >
                {deletingId === confirmDelete.patient_id ? (
                  <>
                    <Loader2 className="spin" size={16} />
                    Usuwanie...
                  </>
                ) : (
                  'Usuń z cache'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {filteredPatients.length === 0 ? (
        <div className="glass-card" style={{ borderLeft: '4px solid #ef4444' }}>
          <p style={{ color: '#aaa', fontSize: '15px' }}>
            {selectedSite !== 'all' ? `Brak zapisanych pacjentów dla Ośrodka ${selectedSite} w cache.` : 'Brak zapisanych pacjentów w cache.'}
          </p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {filteredPatients.map(p => (
            <div 
              key={`${p.site}_${p.patient_id}`} 
              className="glass-card fade-in" 
              style={{ display: 'flex', flexWrap: 'wrap', gap: '20px', alignItems: 'center', justifyContent: 'space-between', padding: '16px 20px', transition: 'transform 0.2s, background-color 0.2s' }}
            >
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', flex: 1, minWidth: '200px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ fontSize: '16px', fontWeight: '700', color: '#fff' }}>📁 Pacjent {p.patient_id}</span>
                  <span style={{ fontSize: '12px', fontWeight: '600', padding: '2px 8px', borderRadius: '12px', backgroundColor: 'rgba(255,255,255,0.05)', color: '#aaa' }}>
                    Ośrodek {p.site}
                  </span>
                  {p.is_complete ? (
                    <span style={{ fontSize: '11px', fontWeight: '600', padding: '2px 8px', borderRadius: '12px', backgroundColor: 'rgba(0,255,0,0.1)', color: '#00ff00', border: '1px solid rgba(0,255,0,0.2)' }}>
                      Kompletny 🟢
                    </span>
                  ) : (
                    <span style={{ fontSize: '11px', fontWeight: '600', padding: '2px 8px', borderRadius: '12px', backgroundColor: 'rgba(245,158,11,0.1)', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.2)' }}>
                      Częściowy 🟡
                    </span>
                  )}
                </div>
                <div style={{ display: 'flex', gap: '16px', fontSize: '13px', color: '#888', marginTop: '4px' }}>
                  <span>Rozmiar: <strong style={{ color: '#aaa' }}>{p.size_mb >= 1024 ? (p.size_mb / 1024).toFixed(2) + ' GB' : p.size_mb + ' MB'}</strong></span>
                  <span>Pliki: <strong style={{ color: '#aaa' }}>{p.file_count}</strong></span>
                  <span>Data cache: <strong style={{ color: '#aaa' }}>{formatDate(p.cached_at)}</strong></span>
                </div>
              </div>

              <div>
                <button 
                  onClick={() => setConfirmDelete(p)}
                  className="btn-secondary"
                  style={{ color: '#ef4444', borderColor: 'rgba(239,68,68,0.2)', backgroundColor: 'rgba(239,68,68,0.02)', padding: '8px 12px', display: 'flex', alignItems: 'center', gap: '6px' }}
                >
                  <Trash2 size={16} />
                  Zwolnij
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
