# PCA Builder & Visualization

Interactive MATLAB tool for learning Principal Component Analysis through hands-on exploration.

![Main Interface](images/shot01.png)

## What is this?

PCA is typically taught with formulas and static diagrams. This tool lets you manipulate parameters and immediately see how correlations affect data structure, how principal components emerge, and why they capture variance in specific directions.

Built for students learning multivariate statistics, educators teaching PCA, or anyone who wants to develop geometric intuition for how PCA works.

## Features

- Interactive control over means, standard deviations, and correlations
- Multiple visualization stages (1D, 2D, 3D, PCA)
- Click-through animations showing geometric decompositions
- Visual projections onto principal component axes
- Built-in educational reference with statistical formulas
- Preset scenarios demonstrating key concepts

![Help System](images/shothelptab.png)

## Installation & Usage

### If you have MATLAB

Download `pca_builder.m` and run it:
```matlab
pca_builder
```

### If you don't have MATLAB

Use MATLAB Online (free, browser-based):

1. Create account at [matlab.mathworks.com](https://matlab.mathworks.com)
2. Upload `pca_builder.m`
3. Run: `pca_builder`

The free tier provides 20 hours/month.

## How to use

Start with the "3x1D" stage to see independent variables, then progress to "2D" to observe correlation effects, "3D" for full data cloud visualization, and "PCA" to see principal components.

Use ROTATE mode to adjust view angles, SELECT mode to click points and deviation lines. The HELP button provides comprehensive documentation.

Try setting all correlations to 0.99 to see data collapse onto a line, or to 0 to see a spherical cloud. The "Try This" tab in the help window has preset scenarios.

## Why this exists

I learned PCA twice - once in second year during independent research, again in third year coursework. The confusion often stems from lacking geometric intuition about variance, standard deviation, and how these concepts extend to multiple dimensions.

This tool won't teach linear algebra from scratch, but aims to make the geometric relationships visual and manipulable. PCA requires thinking in multiple dimensions, which is conceptually challenging. Hopefully this makes it slightly easier.

For deeper understanding of the underlying math, I recommend 3Blue1Brown's linear algebra series on YouTube.

## Citation

If you use this in teaching or research:

**APA:**
```
Bechara, S. (2024). PCA Builder & Visualization: An Interactive Tool 
  for Understanding Principal Component Analysis [Computer software]. 
  https://github.com/sachabechara/pca-visualization-tool
```

**BibTeX:**
```bibtex
@software{bechara2024pca,
  author = {Bechara, Sacha},
  title = {PCA Builder \& Visualization: An Interactive Tool for 
           Understanding Principal Component Analysis},
  year = {2024},
  url = {https://github.com/sachabechara/pca-visualization-tool}
}
```

## Screenshots

<details>
<summary>View all screenshots</summary>

<table>
  <tr>
    <td><img src="images/shot01.png" width="400"/></td>
    <td><img src="images/shot02.png" width="400"/></td>
  </tr>
  <tr>
    <td><img src="images/shot03.png" width="400"/></td>
    <td><img src="images/shot04.png" width="400"/></td>
  </tr>
  <tr>
    <td><img src="images/shot05.png" width="400"/></td>
    <td><img src="images/shot06.png" width="400"/></td>
  </tr>
  <tr>
    <td><img src="images/shot07.png" width="400"/></td>
    <td><img src="images/shot08.png" width="400"/></td>
  </tr>
  <tr>
    <td><img src="images/shot09.png" width="400"/></td>
    <td><img src="images/shot10.png" width="400"/></td>
  </tr>
  <tr>
    <td><img src="images/shot11.png" width="400"/></td>
    <td><img src="images/shot12.png" width="400"/></td>
  </tr>
  <tr>
    <td><img src="images/shot13.png" width="400"/></td>
    <td><img src="images/shot14.png" width="400"/></td>
  </tr>
  <tr>
    <td><img src="images/shot15.png" width="400"/></td>
    <td><img src="images/shot16.png" width="400"/></td>
  </tr>
</table>

</details>

## Contributing

Bug reports, feature requests, and improvements welcome. Open an issue or submit a pull request.

## License

MIT License - free to use and modify. If you improve it, consider contributing back.

## Author

Sacha Bechara  
[github.com/sachabechara](https://github.com/sachabechara)

Built to make PCA more accessible and visually intuitive.
