# UAV-based Visual Remote Sensing for Automated Building Inspection
*Spotlight paper at the <b>Computer Vision for Civil and Infrastructure Engineering (CVCIE) Workshop, ECCV 2022</b>.
### Project: "Drone-based Infrastructure Assessment"
This project aims at accurately detecting and estimating various parameters of a building by deploying an Unmanned Aerial Vehicle (UAV) based remote sensing system. It is a part of [IHub - Mobility Research](https://ihub-data.iiit.ac.in/mobility/) at [IIIT, Hyderabad](https://www.iiit.ac.in/).

#### Objectives:
Through this project, we introduce the following parameters: 
<ol>
<li> Building Plan-shape Estimation </li>
<li> Roof Area Calculation </li>
<li> Distance Between Adjacent Buildings </li>
<li> Roof Layout Estimation </li>
</ol>

For more information, please visit our [project page](https://uvrsabi.github.io/).

#### Data Collection, Installation and Usage:
For details regarding data collection, installation and using the software stack please visit the [documentation website](https://uvrsabi-instructions.readthedocs.io/en/latest/).
<!-- #### Dataset:
We have made our own custom dataset by capturing buildings on IIIT-H campus through a drone. In addition, we have also used the open-source zju_facade dataset to train our models. IIIT-H campus window dataset can be found [here](https://drive.google.com/drive/folders/1fxJP8x9y8I23DFWardhpMM5BZAUY4WtM?usp=sharing).

<p>
  <img src="readme_images/Dataset.drawio.png" width="45%" height="90%"/>     
</p>
For window detection task (The ground truths are bounding boxes shown in red)

<br />
<p>
  <img src="readme_images/RoofTopDataset.png" width="45%" height="90%"/>
</p>

For plan-shape/area (The ground truths are segmented masks in white)

### Progress till now:
Please find our current progress in the following presentation.
<a href="https://docs.google.com/presentation/d/1Oj5h2Y_G0Geoxrf7Ti8xwFlPiwgsrAcYaunN2R8y6zE/edit?usp=sharing" target="_blank">Presentation Link</a>


### Directory Structure:
**win_det_heatmaps:** It contains estimation of window/storey parameters(window detection and post_processing module, window/storey count, storey heights). \
**planShape:** Contains Semantic segmentation and area calculation of roof-tops.

### Window detection
<p align="center"><img src="readme_images/17.png" width="200" height="150"/></p>
<p align = "center">Shufflenet inference</p>
<br />
As shown in the above fig., we have the detected windows from the model inference(Shufflenet from win_det_heatmaps). However, we see that some windows still go undetected. Hence, we have a designed a post-processing module.  

<br />

Post-processing module:
<p align="center"><img src="readme_images/newblockd.drawio (1).png" width="70%" height="40%"/></p>

We take the detected windows as templates and run them over the horizontal patch in the image. We try to match this template in the patch and detect the windows which we were previously not detected. 
<br />


<p align="center"><img src="readme_images/newtm.drawio.png" width=500%" height="70%"/></p>
<p align = "center">Model inference(left), Horizontal Patch
(middle), Template (right)</p>

<p align="center"><img src="readme_images/newio.drawio.png" width=500%" height="70%"/></p>
<p align = "center">Post processing results</p>

As shown in the fig. above, the post processing module detects all windows successfully.
<br />
<br />
<br />

### Storey/Building height estimation:

<p align = "center"><img src="readme_images/VerticalPlaneMapping2.drawio.png" width=50%" height="70%"></p>
As shown in the fig. above, we make use of Depth(D), focal
length of the camera(f), height of the UAV(H) and image
coordinates(x,y) are used to map the coordinates of each
detected window from the image to a 2D vertical plane using
triangulation. 

<br />
<p align="center"><img src="readme_images/nms_vmp.drawio.png" width=70%" height="70%"></p>
<p align="center">2D Vertical Plane Mapping(Before and After NMS)</p>

The above vertical plane helps us get an estimate of distance between 2 consecutive vertical windows. Although we have the imaginary vertical plane(scaled in cm), we cannot use this directly to
estimate storey heights. This is because the vertical plane
also includes the ground plane. Due to this, the estimated
height increases by the proportion of ground plane pixels
and therefore it needs to be accounted for. As it depends
on the start frame and also the camera’s FOV, it is difficult
to generalize it in different scenarios, hence we rely on
3D reconstruction for this.

<p align="center"><img src="readme_images/3d_reconstruction_buildings.png" width=70%" height="70%"></p>

<br />
<p>1 unit (mesh) = ∇Wij /∇wij</p> 
where ∇Wij represents the distance between consecutive
windows in cm, estimated from Plane Mapping Approach
whereas ∇wij represents the distance between same two
windows in the units of mesh from SFM reconstruction

Now, we use the unit scale to estimate the building/storey heights in the 3D reconstruction.
<br />
<br />

### Plan Shape/Area Estimation :
We use RefineNet from [building-footprint-segmentation](https://github.com/fuzailpalnak/building-footprint-segmentation) and fine-tune it on our dataset consisting of GoogleEarth & IIIT-H campus(captured using UAV), which consists of around 200 images.

<p><img src="readme_images/RefineNet_ForwardPass.png" width="100%" height="50%"/></p>
<p align="center">Inference</p>

<p><img src="readme_images/RefineNet_Result-1.drawio.png" width="100%" height="50%"/></p>
<p align="center">Sample results on 4 campus buildings from the dataset - Nilgiri(top-left), Bakul(bottom-left), Aarogya(top-right), Car Service Station(bottom-right)</p>
<br />

Now, we estimate the area(in m²) from the contour Area of the segmented building mask. \
  Area(in m²) = Contour Area(in pixels)*(D/f)² \
  D: Depth(in m)
  f: focalLength(in pixels)
  
<p align="center"><img src="readme_images/rooftoparea_results.png" width=70%" height="70%"></p>

<br />
<br />

### Publications:
-> Dhruv Patel, Shivani Chepuri, Sarvesh Thakur, Harikumar Kandath, Ravi Kiran S, K. Madhava Krishna, “Identifying and estimating salient parameters of a building using UAV based remote sensing”, submitted to IEEE International Conference on Unmanned Aircraft Systems (ICUAS) 2022. 

<br />

### Objectives for next phase:
-> Distance between adjacent buildings \
-> Parapets, objects on roof-top \
-> Staircase exit and water tanks on the roof-top \
-> Cracks on the surface wall and roof-top \
-> Lifelines (electric and water supply, sewage pipes) \
-> Toppling/falling hazard \
-> Building level (flat or tilted ground)


<br /> -->

<!-- ### Project Team:
<ul>
<li>Kushagra Srivastava - Research Assistant, RRC, IIIT Hyderabad </li>
<li>Dhruv Patel - Research Assistant, (RRC), IIIT Hyderabad </li>
<li>
</ul>
Advisors: \
Prof. Madhava Krishna (Head & Professor, RRC, IIIT Hyderabad) \
Dr. Harikumar Kandath (Assistant Professor, IIIT-Hyderabad) \
Dr. Ravi Kiran Sarvadevabhatla (Assistant Professor, IIIT-Hyderabad) -->
