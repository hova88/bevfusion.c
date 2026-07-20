# Publishing the technical article with GitHub Pages

The site under `docs/` is plain HTML, CSS, JavaScript, and inline SVG. It has no
package manager, build step, analytics, remote fonts, or third-party runtime.

## Project site: `owner.github.io/bevfusion.c`

1. Push the repository to GitHub with `docs/index.html` on the default branch.
2. Open **Settings → Pages** in the repository.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Select the default branch and the `/docs` folder, then save.
5. GitHub will show the published URL after the first deployment.

The checked-in `.nojekyll` file tells Pages to serve these files directly.
Every asset uses a relative URL, so forks and project subpaths work without
editing a domain name.

## User or organization site: `owner.github.io`

GitHub requires that site to live in a repository named `owner.github.io`.
Copy the contents of this repository's `docs/` directory to that repository's
root, or keep the project-site URL above. Relative links to repository files
outside `docs/` should be changed to the public source repository URL when the
article is split into a separate repository.

## Local preview

```sh
python3 -m http.server --directory docs 8000
```

Open `http://localhost:8000`. Check a wide desktop viewport and a narrow mobile
viewport. The article honors reduced-motion preferences and remains readable
without JavaScript; JavaScript switches the tensor-stage explainer and filters
the generated checkpoint model explorer. The complete model remains available
as `model-summary.md` when scripting is disabled.
