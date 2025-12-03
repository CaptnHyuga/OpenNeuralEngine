# OpenNeuralEngine Frontend

Modern React-based web interface for OpenNeuralEngine.

## Features

- 🎨 **Beautiful Dark Theme** - Professional look with custom color palette
- ⚡ **Fast** - Built with Vite for instant HMR
- 📱 **Responsive** - Works on desktop and mobile
- 🔄 **Real-time** - WebSocket updates for training progress
- 🧩 **Modular** - Easy to extend with new pages

## Pages

| Page | Description |
|------|-------------|
| Dashboard | Hardware info, quick stats, recent activity |
| Train | Configure and start training runs |
| Inference | Chat interface with model switching |
| Models | Browse local and HuggingFace models |
| Experiments | View and compare training runs |
| Settings | Configure preferences |

## Quick Start

### Development

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

### Production Build

```bash
npm run build
```

Output will be in `dist/` folder.

### With Backend

```bash
# From project root
python launch_web.py --dev
```

This starts both frontend (port 3000) and backend (port 8000).

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **React Router** - Navigation
- **TanStack Query** - Data fetching
- **Zustand** - State management
- **Recharts** - Charts
- **Lucide** - Icons

## Project Structure

```
frontend/
├── src/
│   ├── components/     # Shared UI components
│   │   ├── Layout.tsx
│   │   ├── Sidebar.tsx
│   │   └── Header.tsx
│   ├── pages/          # Page components
│   │   ├── Dashboard.tsx
│   │   ├── Train.tsx
│   │   ├── Inference.tsx
│   │   ├── Models.tsx
│   │   ├── Experiments.tsx
│   │   └── Settings.tsx
│   ├── lib/            # Utilities
│   │   └── utils.ts
│   ├── api.ts          # API client
│   ├── store.ts        # Global state
│   ├── App.tsx         # Root component
│   ├── main.tsx        # Entry point
│   └── index.css       # Global styles
├── public/             # Static assets
├── index.html          # HTML template
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json
```

## API Integration

The frontend communicates with the backend through:

- **REST API** - `/api/*` endpoints
- **WebSocket** - `/ws` for real-time updates

All API calls go through `src/api.ts`.

## Customization

### Colors

Edit `tailwind.config.js` to change the color palette:

```js
colors: {
  primary: { ... },  // Blue tones
  accent: { ... },   // Purple tones
  surface: { ... },  // Gray tones
}
```

### Adding a Page

1. Create component in `src/pages/`
2. Add route in `src/App.tsx`
3. Add nav item in `src/components/Sidebar.tsx`

## License

MIT
