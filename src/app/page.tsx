export default function HomePage() {
  return (
    <main className="min-h-screen bg-[linear-gradient(180deg,#F8FAFC_0%,#EEF4FF_100%)] px-6 py-10 text-slate-900">
      <section className="mx-auto flex w-full max-w-6xl flex-col gap-8 rounded-[2rem] bg-white/85 p-8 shadow-[0_30px_80px_rgba(15,23,42,0.12)] ring-1 ring-slate-200 backdrop-blur">
        <div className="flex flex-col gap-4 rounded-[1.75rem] bg-[linear-gradient(135deg,#2563EB_0%,#0F172A_100%)] p-8 text-white">
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-blue-100">StorePulse AI</p>
          <h1 className="text-4xl font-black tracking-tight sm:text-5xl">Turning Store Feedback Into Intelligent Action</h1>
          <p className="max-w-3xl text-base text-blue-50/90 sm:text-lg">
            AI-powered retail intelligence and early-warning platform for multilingual customer reviews, staff notes,
            and complaint transcripts.
          </p>
          <div className="flex flex-wrap gap-3 text-sm font-medium">
            <span className="rounded-full bg-white/12 px-4 py-2">Policy-first safety escalation</span>
            <span className="rounded-full bg-white/12 px-4 py-2">Emerging incident detection</span>
            <span className="rounded-full bg-white/12 px-4 py-2">Weekly regional action reports</span>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {[
            ["Multilingual Analyzer", "English, Hinglish, Telugu/Tamil/Marathi transliterations"],
            ["Executive Dashboard", "KPI cards, trends, workload, and filters"],
            ["Early Warning System", "Store-level normalized-rate spike detection"],
            ["Evidence & Reporting", "Quotes, routing, PDF/CSV exports"],
          ].map(([title, description]) => (
            <article key={title} className="rounded-[1.5rem] border border-slate-200 bg-slate-50 p-5 shadow-sm">
              <h2 className="text-lg font-semibold text-slate-950">{title}</h2>
              <p className="mt-2 text-sm leading-6 text-slate-600">{description}</p>
            </article>
          ))}
        </div>

        <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          <section className="rounded-[1.5rem] border border-slate-200 bg-white p-6">
            <h2 className="text-xl font-bold text-slate-950">How to run the local demo</h2>
            <ol className="mt-4 space-y-3 text-sm leading-6 text-slate-700">
              <li>1. Install Python dependencies: <code className="rounded bg-slate-100 px-2 py-1">pip install -r requirements.txt</code></li>
              <li>2. Launch the app: <code className="rounded bg-slate-100 px-2 py-1">streamlit run app.py</code></li>
              <li>3. Use the sidebar to switch between dashboard, analyzer, alerts, reports, and evaluation.</li>
            </ol>
            <p className="mt-4 text-sm text-slate-500">
              The Streamlit app uses synthetic demonstration data by default and supports official CSV uploads.
            </p>
          </section>

          <aside className="rounded-[1.5rem] border border-slate-200 bg-slate-50 p-6">
            <h2 className="text-xl font-bold text-slate-950">Hackathon Highlights</h2>
            <ul className="mt-4 space-y-3 text-sm leading-6 text-slate-700">
              <li>• 2,000+ synthetic multilingual feedback records</li>
              <li>• Safety escalation for allergen, hygiene, and misconduct issues</li>
              <li>• Duplicate screening across channels</li>
              <li>• PDF and CSV action brief exports</li>
              <li>• SQLite-backed local storage</li>
            </ul>
          </aside>
        </div>
      </section>
    </main>
  );
}
