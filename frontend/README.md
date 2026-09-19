# Agent UI build

`npm ci && npm run build` renders shadcn/ui Button, Card, Badge and Input
components into `src/agent/web/index.html` and compiles `shadcn.css`.
Edit `frontend/index.template.html`, not the generated index.html.
Existing app.js owns all robot interactions; no React hydration or CDN runs
in the browser. Runtime servers do not require Node.

Component source: https://ui.shadcn.com/r/styles/new-york-v4/{button,card,badge,input}.json
Upstream: https://github.com/shadcn-ui/ui (MIT).
Imports are adapted to the local utils and Radix Slot package.
The stylesheet preserves semantic red for STOP/errors and camera detection colours.

Browser regression: `tests/keyboard-check.py` uses Playwright against a dedicated
hardware-free preview on localhost:8110 and intercepts all motion/STOP requests.
It checks every keyboard binding, repeat/typing/busy guards, blur disarming,
Escape, component markup, neutral primary colour and mobile overflow.
Do not point the preview fixture at a real robot session.

Theme: the header button toggles light/dark and stores `so101-theme` in
localStorage. With no explicit preference, the OS colour scheme is used and
followed. The head script sets the theme before paint. `tests/theme-check.py`
verifies system default, override, reload persistence, popover colours and mobile width.
