#import("@preview/bubble:0.1.0"): *
#import "@preview/subpar:0.1.1"
#import "@local/codelst:2.0.2": sourcecode

#show: it => bubble(
  title: "Soft object in MuJoCo",
  subtitle: "Creation of a deformable object from a 3D mesh",
  author: "Youenn LE JEUNE",
  affiliation: "INSA Rennes / DIAG Sapienza",
  date: datetime.today().display(),
  year: none,
  class: none,
  logo: {
    image("../assets/logo-insa.png")
    image("../assets/logo-sapienza.png")
  },
  it
)
#set text(lang: "en")

#set heading(numbering: "Step 1.1:")

#let rarr = sym.arrow


= Prepare the 3D model
== Import the model
To begin with, you need to have a 3D model file to use. It can be of any filetype: we will import it in Blender, which supports many 3D model filetypes.

Once you have your model, open Blender. You should see something similar with the @blender-welcome.
#figure(image("assets/blender-welcome.png", width: 85%), caption: [Blender welcome screen]) <blender-welcome>

Press _escape_ to leave the welcome screen. Select the "Cube" from the right panel and press _delete_. Now you can import the model: go to "File", "Import", choose the right format and then import your file.

== Remove impurities
Some models may not be directly suitable to convert to MuJoCo. For instance, if they contain holes in their surface (@imp-liver) or voids inside their volume (@imp-kidney), they must be fixed.

#subpar.grid(
  figure(image("assets/impure-kidney.png"), caption: "Kidney") ,<imp-kidney>,
  figure(image("assets/impure-liver.png"), caption: "Liver"), <imp-liver>,
  columns: (1fr, 1fr),
  caption: [Impure models]
)

== Simplify mesh
Most 3D models are way too detailed to be imported in MuJoCo: it would cause the simulation to be extremely slow and require a gigantic amount of memory to run. Therefore, we must make our models "low-poly" by reducing the amount of vertices.

To do that, in Object Mode, select your mesh in the right panel. Go in the "Modifiers" menu at the bottom right and add a "Decimate" modifier (in the "Generate" submenu). See @modifier-add.

#subpar.grid(
  figure(image("assets/decimate-modifier-add.png", width: 95%), caption: "Creation of the modifier"), <modifier-add>,
  figure(image("assets/decimate-modifier-params.png", width: 85%), caption: "Parameters"), <modifier-params>,
  columns: (1fr, .5fr),
  numbering-sub: "1a",
  show-sub-caption: (num, it) => [#it.supplement #num: #it.body],
  align: top
)

Once added, you can edit the "ratio" parameter (see @modifier-params). The goal is to have a low face count value (below 400 from my experience). Lowering the ratio will collapse vertices, converting the high-resolution model to a low-resolution one.

#subpar.grid(
  figure(image("assets/high-res.png", width: 60%), caption: "High resolution"),
  figure(image("assets/low-res.png", width: 60%), caption: "Low resolution"),
  columns: (1fr, 1fr),
  caption: "Simplification of the liver model"
)

== Export the mesh
Once you are satisfied with the mesh simplification, you can export it. To do that, in Object Mode, select the mesh in the right panel. Then go to "File", "Export" and choose #box["STL (.stl)"]. In the options at the right side of the export window, tick "Selected Only". Export your mesh at the path of your choice.

= Conversion to volumetric mesh #footnote[Informations have been gathered from #link("https://github.com/google-deepmind/mujoco/issues/1724#issuecomment-2161135394")[this GitHub issue].]
== Convert your mesh #footnote[This part is a transcription of #link("https://www.youtube.com/watch?v=C5BWRWCr7Ck")[this YouTube video].]
In Gmsh (installable for free on #link("https://gmsh.info/")), open your STL file using "File", "Open...". You should see a window similar to @gmsh-initial.

#figure(image("assets/gmsh-initial.png", width: 70%), caption: "GMSH window after opening") <gmsh-initial>

In the left panel, click on "Modules" #rarr "Geometry" #rarr "Elementary Entities" #rarr "Add" #rarr "Volume", then click on an edge of your mesh. You may get a popup asking you to create a .geo file #sym.dash click on "Proceed as is", then press on _e_ to end the selection.

Now click on "Modules" #rarr "Mesh" #rarr "3D" to create a volumetric mesh. If you get an error at this point, see @gmsh-error.

Once you have a valid 3D mesh, go to "File" #rarr "Export". Choose `Mesh - Gmsh MSH (.msh)` as the format and save. When it asks you for specific msh format, choose `Version 4 ASCII`.

== Help! "3D" gives me errors <gmsh-error>
At the step of clicking on "3D" in Gmsh, you might get errors. This usually means the original mesh exported from Blender contains impurities or malformed sections. This must be resolved by hand on an individual basis.

See @gmsh-issue for an example of an issue. It was resolved by looking at the location specified by the blue sphere in Blender: there was a stray vertex creating an erroneous face. After deleting this face in Edit Mode, the 3D operation went successsfully.
#subpar.grid(
  figure(image("assets/gmsh-issue.png"), caption: "Mesh issue"),
  [
    #figure(image("assets/gmsh-error.png"), caption: "Full error")
    #text(hyphenate: false)[(This kind of full error is accessible by clicking on the bar at the bottom of the Gmsh window.)]
  ],
  caption: "Example of a Gmsh issue when 3D meshing",
  label: <gmsh-issue>,
  columns: (1fr, 1fr),
  align: top
)

== Clean the mesh
Your mesh file contains the necessary data and even more! Unfortunately, MuJoCo requires the .msh file to require _only_ the necessary data (the volume and its associated tetrahedron). Therefore, we must clean the mesh of the unnecessary data.

Fortunately, someone created a Python script which does exactly that: https://github.com/mohammad200h/GMSHConverter.
Clone it or download the "tree" branch, install the required libraries and run it using the following command:

```sh
$ python3 gmsh_cleaner.py -v 4.1 -i '<file>.msh' -o '<file>_converted.msh'
```

This will create many files, including `<file>_converted_vol.msh`. That is the important one, save it for later.

= Importing in MuJoCo
Now we have a file containing the necessary data for MuJoCo to create the soft body, we can include this file in an existing MJCF .xml file. There are multiple things to add in the file:

== Model parameters
In order to make a soft body work, there are some parameters to adjust in the whole model:
```xml
<option solver="CG" tolerance="1e-6" timestep=".001" integrator="implicitfast" jacobian="sparse"/>

<size memory="100M"/>

<extension>
    <plugin plugin="mujoco.elasticity.solid" />
</extension>
```
The `option` tags seems to be necessary to have a stable soft simulation. The `timestep` can be set to be a little faster but not by much, 0.001 is a good start.

== Soft body
The most important thing to add is the actual declaration of the soft body. You will put it wherever you want in your bodies hierarchy, in the `<worldbody>` section:
```xml
<flexcomp name="soft_body_name" pos="x y z" type="gmsh" dim="3"
    file="file.msh" scale="1 1 1">
  <edge equality="true"/>
  <pin id="47 38 58"/>
  <contact internal="false" solref="0.005 1" solimp=".95 .99 .0001" selfcollide="none" />
  <plugin plugin="mujoco.elasticity.solid">
      <config key="poisson" value="0.1" />
      <config key="young" value="5e4" />
      <config key="damping" value="0.005" />
  </plugin>
</flexcomp>
```
Among the parameters you can edit:
- the `pos` and `scale` attributes of `<flexcomp>`.
- `<pin />` to choose which bodies to pin in the world. The IDs are not guessable: use the MuJoCo "Simulate" executable, show the body names in the "Rendering" menu and choose which bodies to pin.
- the parameters of `<contact />`, which are not easily understandable. See MuJoCo XML reference.
- the 3 `<config />` values. 

Finally, if you want to add a material or texture to the soft body, you'll have to create it in the `<asset>` section of your model and assign it in the `<flexcomp>`.