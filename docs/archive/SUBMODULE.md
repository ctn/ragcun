# LeJEPA Submodule

The official LeJEPA implementation is included as a git submodule at `external/lejepa`.

## What is a Git Submodule?

A git submodule allows you to:
- Track a specific version of LeJEPA
- Access the source code locally
- Update to new versions when needed
- Keep your project and LeJEPA separate

## Location

```
/Users/ctn/src/ctn/ragcun/external/lejepa/
```

This points to: `git@github.com:rbalestr-lab/lejepa.git`

## Initial Setup (Already Done)

The submodule has been added with:

```bash
git submodule add git@github.com:rbalestr-lab/lejepa.git external/lejepa
```

## For New Clones

If someone else clones your ragcun repository, they need to initialize the submodule:

```bash
git clone git@github.com:ctn/ragcun.git
cd ragcun

# Initialize and fetch submodule
git submodule init
git submodule update

# Or do it in one command:
git submodule update --init --recursive
```

## Installation

The submodule is automatically installed when you run:

```bash
pip install -e .
```

This is because `requirements.txt` includes:
```
-e external/lejepa
```

## Updating LeJEPA

To update to the latest version of LeJEPA:

```bash
cd external/lejepa
git pull origin main
cd ../..

# Commit the updated submodule reference
git add external/lejepa
git commit -m "Update LeJEPA submodule to latest version"
```

## Checking Current Version

```bash
# See which commit the submodule is at
git submodule status

# Or go into the submodule
cd external/lejepa
git log -1
git describe --tags
```

## Benefits of Using Submodule

### ✅ Advantages:

1. **Version Control**: Know exactly which LeJEPA version you're using
2. **Source Access**: Can read and debug LeJEPA code locally
3. **No Network Needed**: Once cloned, works offline
4. **Reproducibility**: Pin to specific commit for reproducible results

### ⚠️ Alternative (If Submodule Causes Issues):

If you prefer, you can still install from GitHub directly:

```bash
# Remove from requirements.txt: -e external/lejepa
# Add instead:
pip install git+https://github.com/rbalestr-lab/lejepa.git
```

## Common Submodule Commands

### Clone ragcun with submodule:
```bash
git clone --recurse-submodules git@github.com:ctn/ragcun.git
```

### Update submodule to latest:
```bash
git submodule update --remote external/lejepa
```

### Check submodule status:
```bash
git submodule status
```

### Go to specific LeJEPA commit:
```bash
cd external/lejepa
git checkout <commit-hash>
cd ../..
git add external/lejepa
git commit -m "Pin LeJEPA to specific version"
```

## Directory Structure

```
ragcun/
├── external/              # External dependencies
│   └── lejepa/           # LeJEPA submodule (git@github.com:rbalestr-lab/lejepa.git)
│       ├── lejepa/       # Python package
│       │   ├── univariate/
│       │   └── multivariate/
│       ├── tests/
│       ├── scripts/
│       └── README.md
├── ragcun/               # Your code
├── notebooks/            # Your notebooks
└── .gitmodules           # Submodule configuration
```

## Troubleshooting

### Submodule directory is empty?

```bash
git submodule update --init
```

### Want to remove submodule?

```bash
# Don't do this unless you really want to remove it
git submodule deinit external/lejepa
git rm external/lejepa
rm -rf .git/modules/external/lejepa
```

### Submodule conflicts after git pull?

```bash
git submodule update --init --recursive
```

## Using LeJEPA in Code

Since it's installed via requirements.txt, you can import normally:

```python
import lejepa

# Use LeJEPA
sigreg = lejepa.multivariate.SlicingUnivariateTest(
    univariate_test=lejepa.univariate.EppsPulley(num_points=17),
    num_slices=1024
)
```

## Notes

- The submodule tracks a specific commit (currently latest)
- When you `git pull` ragcun, the submodule doesn't auto-update
- Run `git submodule update` to sync to the commit tracked by ragcun
- Or `git submodule update --remote` to get the absolute latest from LeJEPA

## References

- LeJEPA Repository: https://github.com/rbalestr-lab/lejepa
- Git Submodules Documentation: https://git-scm.com/book/en/v2/Git-Tools-Submodules
