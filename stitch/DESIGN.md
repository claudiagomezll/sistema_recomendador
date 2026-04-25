# Design System Document: High-End Editorial Dark Mode

## 1. Overview & Creative North Star: "The Technical Architect"
This design system is built to transform complex configuration into a premium, editorial experience. The Creative North Star is **"The Technical Architect"**—a philosophy that balances the precision of an expert-mode tool with the aesthetic depth of a luxury digital interface.

We move beyond the "flat dashboard" trope by utilizing **Atmospheric Depth**. Instead of rigid grids and 1px borders that clutter the eye, we use intentional asymmetry, layered tonal surfaces, and high-contrast typography scales. The goal is to make the user feel like they are interacting with a sophisticated instrument rather than a standard web form.

---

## 2. Colors: Tonal Depth & Vibrant Accents
Our palette is rooted in a deep navy foundation, using vibrant blue and cyan accents to draw the eye to critical actions and data points.

### The "No-Line" Rule
**Explicit Instruction:** Designers are prohibited from using 1px solid borders to section off major UI areas. Boundaries must be defined through background color shifts. Use `surface_container_low` for sections and `surface_container_highest` for nested elements. This creates a "molded" look rather than a "sketched" look.

### Surface Hierarchy & Nesting
Treat the UI as a physical stack of materials:
*   **Base Layer:** `surface` (#0f131f) – The infinite void.
*   **Section Layer:** `surface_container_low` (#171b28) – Major content groupings.
*   **Interaction Layer:** `surface_container_high` (#262a37) – Input areas and secondary cards.
*   **Floating Layer:** `surface_bright` (#353946) – Tooltips and active modals.

### The "Glass & Gradient" Rule
To elevate the "Expert Mode" feel, use **Glassmorphism** for floating elements (Modo Simple toggle, tooltips). Apply `surface_container` with a `backdrop-blur: 12px` and 60% opacity. 
*   **Signature Textures:** Main CTAs (e.g., "Generar Propuesta") must use a linear gradient from `primary_container` (#2962ff) to `secondary_container` (#0231de) at a 135-degree angle to provide a sense of energy and motion.

---

## 3. Typography: The Editorial Contrast
We use a dual-font strategy to balance technical utility with premium character.

*   **Display & Headlines (Plus Jakarta Sans):** These are our "Brand" moments. Use `display-md` and `headline-lg` with tight letter-spacing (-0.02em) to create an authoritative, architectural feel.
*   **Body & Labels (Manrope):** Chosen for its high legibility in dense configuration environments. `label-md` is the workhorse for input headers, while `body-sm` handles helper text.
*   **Hierarchy Note:** Always maintain a significant scale jump between section headers (`headline-sm`) and field labels (`label-md`). This visual "gap" helps the user scan complex forms without cognitive overload.

---

## 4. Elevation & Depth: Tonal Layering
Traditional shadows are too "heavy" for a technical dark theme. We achieve lift through light.

*   **The Layering Principle:** Place a `surface_container_lowest` (#0a0e1a) card inside a `surface_container_low` (#171b28) section to create a "recessed" effect for inputs, making them feel like physical slots in a console.
*   **Ambient Shadows:** For floating menus, use a shadow with a 40px blur and 8% opacity, using the `primary` (#b6c4ff) color as the shadow tint. This mimics a subtle glow rather than a dark stain.
*   **The "Ghost Border" Fallback:** If a container requires a border for accessibility, use `outline_variant` at **15% opacity**. It should be felt, not seen.

---

## 5. Components: Precision Elements

### Input Fields
*   **Container:** Use `surface_container_highest` with a `DEFAULT` (0.5rem) corner radius.
*   **States:** On focus, transition the border from "Ghost" to `tertiary` (#00daf3) at 100% opacity. This neon-like glow signals "Expert" precision.
*   **Text:** Placeholder text must use `on_surface_variant`.

### Buttons
*   **Primary:** Gradient-filled (see Glass & Gradient Rule) with white `on_primary_container` text. Use `xl` (1.5rem) corner radius for a modern, pill-shaped silhouette.
*   **Secondary (Modo Simple):** Use a `surface_bright` container with a subtle `outline`. This should feel like a toggle on a high-end stereo.

### Cards & Grouping
*   **Forbid Dividers:** Do not use horizontal rules (`<hr>`). Use 48px or 64px of vertical white space (from the spacing scale) to separate major logical blocks.
*   **Visual Grouping:** Use subtle background shifts. A group of related inputs should sit on a `surface_container_low` base that is slightly lighter than the app background.

### Chips
*   **Action Chips:** Use `secondary_fixed_dim` with `on_secondary_fixed` text for high-contrast interactive tags.

---

## 6. Do's and Don'ts

### Do
*   **DO** use `tertiary` (#00daf3) for success states and secondary accents to keep the "Technical" vibe alive.
*   **DO** leave generous breathing room. Complexity in the form requires simplicity in the layout.
*   **DO** use `backdrop-blur` on navigation bars and floating toggles to maintain a sense of depth.

### Don't
*   **DON'T** use pure black (#000000). Always use the `surface` (#0f131f) deep navy to ensure the "Editorial" depth is maintained.
*   **DON'T** use 100% white for body text. Use `on_surface_variant` (#c3c5d8) to reduce eye strain in dark mode.
*   **DON'T** use sharp 90-degree corners. Everything should have at least a `sm` (0.25rem) radius to feel "Accessible."
*   **DON'T** stack more than three levels of surface containers. Too much nesting leads to visual "noise."