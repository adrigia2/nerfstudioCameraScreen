<br/>

```{image} _static/imgs/logo.png
:width: 400
:align: center
:alt: nerfstudio
:class: only-light
```

```{image} _static/imgs/logo-dark.png
:width: 400
:align: center
:alt: nerfstudio
:class: only-dark
```

<br/>

Nerfstudio provides a simple API that allows for a simplified end-to-end process of creating, training, and visualizing NeRFs.
The library supports an **interpretable implementation of NeRFs by modularizing each component.**
With modular NeRF components, we hope to create a user-friendly experience in exploring the technology.
Nerfstudio is a contributer friendly repo with the goal of building a community where users can easily build upon each other's contributions.

It's as simple as plug and play with nerfstudio!

Ontop of our API, we are commited to providing learning resources to help you understand the basics of (if you're just getting start), and keep up-to-date with (if you're a seasoned veteran) all things NeRF.
As researchers, we know just how hard it is to get onboarded with this next-gen technology. So we're here to help with tutorials, documentation, and more!

Finally, have feature requests? Want to add your brand-spankin'-new NeRF model? Have a new dataset? **We welcome any and all [contributions](reference/contributing)!**
Please do not hesitate to reach out to the nerfstudio team with any questions via [Discord](https://discord.gg/NHGtYRAW).

We hope nerfstudio enables you to build faster 🔨 learn together 📚 and contribute to our NeRF community 💖.

## Contents

```{toctree}
:hidden:
:caption: Getting Started

quickstart/installation
quickstart/first_nerf
quickstart/custom_dataset
quickstart/viewer_quickstart
Contributing<reference/contributing>
```

```{toctree}
:hidden:
:caption: NeRFology

nerfology/methods/index
nerfology/model_components/index
```

```{toctree}
:hidden:
:caption: Developer Guides

developer_guides/pipelines/index
developer_guides/viewer/index
developer_guides/config
developer_guides/debugging_tools/index
```

```{toctree}
:hidden:
:caption: Reference

reference/cli/index
reference/api/index
```

This documentation is organized into 3 parts:

- **🏃‍♀️ Getting Started**: a great place to start if you are new to nerfstudio. Contains a quick tour, installation, and an overview of the core structures that will allow you to get up and running with nerfstudio.
- **🧪 Nerfology**: want to learn more about the tech itself? We're here to help with our educational guides. We've provided some interactive notebooks that walk you through what each component is all about.
- **🤓 Developer Guides**: describes all of the components and additional support we provide to help you construct, train, and debug your NeRFs. Learn how to set up a model pipeline, use the viewer, create a custom config, and more.
- **📚 Reference**: describes each class and function. Develop a better understanding of the core of our technology and terminology. This section includes descriptions of each module and component in the codebase.

## Supported Methods

- [**Nerfacto**](https://github.com/nerfstudio-project/nerfstudio/blob/master/nerfstudio/models/nerfacto.py): our defacto NeRF method combines modules focused on quality with modules focused on faster rendering. Nerfstudio easily lets us experiment with the best of both worlds!
- [NeRF](https://www.matthewtancik.com/nerf): Representing Scenes as Neural Radiance Fields for View Synthesis
- [Instant NGP](https://nvlabs.github.io/instant-ngp/): Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
- [Mipnerf](https://jonbarron.info/mipnerf/): A Multiscale Representation for Anti-Aliasing Neural Radiance Fields
- [NerfW](https://nerf-w.github.io/): Neural Radiance Fields for Unconstrained Photo Collections
- [Semantic NeRF](https://shuaifengzhi.com/Semantic-NeRF/): In-Place Scene Labelling and Understanding with Implicit Scene Representation
- [TensoRF](https://apchenstu.github.io/TensoRF/): Tensorial Radiance Fields

We'll be constantly growing this list! So make sure to check back in to see our updates.

**Eager to contribute?** We'd love to see you use nerfstudio in implementing new (or even existing) methods! Feel free to contact us directly or view our [Contributor's Guide](reference/contributing) to see how you can get your model on this list!

## Quicklinks

|                                                            |                        |
| ---------------------------------------------------------- | ---------------------- |
| [Github](https://github.com/nerfstudio-project/nerfstudio) | Official Github Repo   |
| [Discord](https://discord.com/invite/NHGtYRAW)             | Join Discord Community |
| [Viewer](https://viewer.nerf.studio/)                      | Web-based Nerf Viewer  |

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`