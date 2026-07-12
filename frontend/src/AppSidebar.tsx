import { Network, PanelsTopLeft } from 'lucide-react'


export type AppView = 'workspace' | 'graph'


type AppSidebarProps = {
  activeView: AppView
  onChange: (view: AppView) => void
}


const NAV_ITEMS: Array<{
  id: AppView
  label: string
  icon: typeof PanelsTopLeft
}> = [
  { id: 'workspace', label: 'Workspace', icon: PanelsTopLeft },
  { id: 'graph', label: 'Knowledge graph', icon: Network },
]


export function AppSidebar({ activeView, onChange }: AppSidebarProps) {
  return (
    <nav className="app-sidebar" aria-label="Application views">
      <div className="app-sidebar-brand" aria-label="Video Course Cards">
        VC
      </div>
      <div className="app-sidebar-nav">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon

          return (
            <button
              key={item.id}
              type="button"
              className={activeView === item.id ? 'active' : ''}
              aria-label={item.label}
              aria-current={activeView === item.id ? 'page' : undefined}
              title={item.label}
              onClick={() => onChange(item.id)}
            >
              <Icon aria-hidden="true" size={20} strokeWidth={1.8} />
              <span>{item.label}</span>
            </button>
          )
        })}
      </div>
    </nav>
  )
}
