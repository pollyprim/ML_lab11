https://www.kaggle.com/t/6483b088799a46bd8fc8fbd377ef86a4
Overview
One of the hardest problem in modern ML world is to create a system that can handle different modalities by understanding their relationships as well as properties. Such systems are becoming prevalent in different areas, especially when we are trying to solve such problems as:

How to find a specific entity using just its text description and vice versa?
In real-world problem we can expand such kind of tasks

How to recommend\find that has only visual representation of something based only on a text data?
etc.
In order to find a solution we need a system that could handle both modalities that help in understanding their properties simultaneously. You are proposed to solve a task where images have some logical statements that explain relationship between objects on the image. Your goal is to construct such a model that would assess those statements as correct as possible. Each image shows a 3D scene with simple objects. Each caption is a statement about the scene. You need to predict the probability (or score) that the caption is true for the image. Captions range from simple ("a red cube") to compositional ("there are exactly two objects behind the green sphere").

The most straightforward method is to separately train two encoders for two modalities separately that will be shown for during the lab. Briefly it might look as the following:

image   ──► image encoder ──► i (256 numbers)
caption ──► text encoder  ──► t (256 numbers)
score = scale · cos(i, t) + bias  ──► sigmoid ──► probability that the caption is true
Still you can find different ways to solve that problem. In this competition, you need to design such a system that can assess logical statements for 3D shapes in images.

Description
You are provided a training dataset to design your own model (solution) and test set to assess an ROC-AUC score. You have following structure of the dataset of image-caption pairs:

train - the training set (5000 pairs)
val - the validation set (1000 pairs)
test - the test set (6000 pairs)
Also testing subset is divided into public and private parts in ratio of 70% and 30% accordingly. Public part is needed to calculate your metric for a submission file during the competition time, while private is needed to get a final score.

Each pair in the dataset contains an image id, caption and type of logical statement. Overall there are 5 different object classes of different context in the dataset. This dataset involves the following statement types:

attribute
counting
binding
negation
position
spatial
spatial_counting
spatial_negation
comparison
You might notice that statement types sometimes could be about a single object, while others could describe relationship between objects in the image.

Evaluation
Submissions are evaluated on area under the ROC curve between the predicted probability or score by your model and the observed target.

Submission File
For every id in test.csv, submit a label: a score saying how likely the caption is TRUE for the image. Higher score = more likely TRUE. The score can be a probability of TRUE (0…1) or any other real number (a logit, a cosine similarity), because ROC-AUC only uses the order of the scores. Do not submit the probability of FALSE: your AUC would become 1 − AUC. Do not submit hard 0/1 predictions either: they are accepted, but every row with the same value ties, and you lose most of the ranking information.

 id,label
 11396,0.87
 11397,0.12
Please take attention that this time IT IS NOT an image id, but a sample id.

Dataset Description
Each image shows a 3D scene with simple objects (cubes, spheres, cylinders; 8 colours, 2 sizes, 2 materials: rubber or metal). Each caption is a statement about the scene that is either TRUE or FALSE for its image. Your task is to predict the probability that a caption is true.

Files
train/, val/, test/ - images (240x160 JPEG). An image listed in train.csv is in train/, and so on.
train.csv, val.csv - captions with labels.
test.csv - captions without labels: predict these.
sample_submission.csv - a submission in the correct format (id, label) - (id of sample, prob).
Split	Images	Captions
train	5,000	
val	1,000	<…>
test	6,000	<…>
Columns
id - unique row id (used in the submission)
image - image file name
caption - the statement about the image
level - difficulty level of the caption type, 1 (simple) to 5 (compositional)
category - caption type (see below)
pair_id - (train/val only) links a caption to its "twin": a caption for the same image that differs in one detail (a colour, a number, left/right, "a"/"no", …). Exactly one caption of each pair is true.
label - (train/val only) 1 = the caption is true for the image, 0 = false
test.csv has id, image, caption, level, category (no pair_id, no label).

Caption categories
Level	Category	Example
1	attribute	a large red metal cube
2	counting	there are exactly three cylinders
3	binding	a red cube and a blue sphere
3	negation	there is no gray cylinder
4	position	the red cube is on the left side of the image
4	spatial	a red cube to the left of the blue sphere
5	spatial_counting	there are exactly two objects behind the green sphere
5	spatial_negation	there is no sphere to the right of the red cube
5	comparison	there are more cubes than spheres
Labels are balanced: 50% true, 50% false in every category. There are some of the test captions use phrasings (word orders) that DO NOT appear in train or val.

Competition Rules
By participating, you agree to these rules, the course academic-integrity policy, the current Rating Statement, and Kaggle’s Terms of Service.

2. Individual checkpoint
Each student starts the individual checkpoint during their assigned laboratory class on November 12-14, uploads the notebook and results by the end of that class, and defends the work during their assigned class on November 19-21.

For the individual checkpoint, every student must:

attend their assigned laboratory class;
independently finish a solution model by completing lab assignments providing an AUC-ROC score as the result;
submit the required notebook and results through their own Moodle account;
be able to explain and reproduce the solution.
General discussion is permitted, but students must not exchange code or prediction files during the individual phase.

Submitting the individual work is required before joining a team or entering the team phase. Checkpoint credit and leaderboard-bonus eligibility also require a successful individual defense the following week.

3. Teams
After submitting their individual checkpoint, students may continue alone or form teams of up to three students.

Every member must be enrolled in the course, have submitted the individual checkpoint, and complete its defense for eligibility.
Teams must be finalized in Kaggle by the merger deadline.
The final Moodle team list must exactly match the Kaggle team membership.
Team membership is final after the merger deadline.
Collaboration is permitted only within the declared team.
Code, engineered features, predictions and private competition materials must not be exchanged between teams.
Competition data and invitation links must not be shared outside the course or posted publicly.
Every member must make a meaningful contribution and be able to explain the complete solution.
Guest, ghost, gifted or honorary membership is prohibited.
The organizers may assess individual understanding separately. A student who cannot explain the submitted work may lose individual credit even if the team submission remains valid.

4. ALLOWED methods
Participants may use standard CPU or GPU-compatible machine learning methods for Deep Learning. You can go through it using Pytorch or TensorFlow. Our Lab solutions will be based on Pytorch methods.

In order to design a model you are allowed to use two kind of architectures that transform a specific modality into a vector. In practice, it insists to use two type of models: image-based and text-based architectures, then combine their results as a source of information about described entities.

Models that are ALLOWED to use:

Image encoders: classification models from the torchvision list (https://docs.pytorch.org/vision/0.22/models.html) with their published ImageNet weights, frozen or fine-tuned. You may use any part of them (e.g. intermediate feature maps).
BUT: pretrained detection, segmentation, video and optical-flow models from that list (e.g. Faster R-CNN, Mask R-CNN, DeepLabV3) ARE NOT ALLOWED: their weights were trained to find or outline objects on other datasets.
Text encoders: any text-only model that returns vector representations: RNN / LSTM / GRU, or pretrained encoder-only Transformers (BERT, etc.), run locally.
CLIP / SigLIP / OpenCLIP dual encoders are allowed (zero-shot, fine-tuned or as feature extractors): each modality is encoded separately.
Any layers you design and train yourself on the contest data ARE ALLOWED: fusion (e.g. FiLM), attention, layers that use positions in the feature map, and the final layers that predict whether a caption is true.
5. NOT ALLOWED methods and models:
Pretrained models that process images and text together or generate one from the other.
Example of architectures: LLaVA, Qwen-VL, GPT-4o, Gemini, BLIP / BLIP-2, ALBEF, ViLT, FLAVA, OWL-ViT, Grounding DINO, captioning and VQA models, or similar.
LLMs or other text generators, including for creating extra captions.
External data of similar type (images, scene files, questions) as well as models trained on similar datasets. So carefully read how the models were trained and on which datasets before applying them to your solution.
Any methods that are trying to match contest images to external images.
Manual labelling of test images, and any information about hidden labels.
Pre-fitted models or predictions produced outside the declared team.
BUT: with the given data you are ALLOWED to do the following:

training on train.csv AND val.csv (e.g. a final model trained on both);
ensembles: combining predictions of several of your own models (e.g. averaging, or different models for different categories);
data augmentation, and extra captions created from the given training data;
using the unlabelled test images and captions (e.g. finding twin captions).
As for the augmentation part, you should be careful, as NOT every method can be used straightly with the dataset. Why?

Augmentation type	Safe?	Reason
Horizontal flip	Only if the caption is changed too	“left” and “right” must be swapped in the caption, otherwise the label becomes wrong
Vertical flip, rotation	No	Breaks “in front of” / “behind” and makes scenes physically impossible
Random crop, zoom-in	Risky	An object at the edge can be cut off, so counts and “there is…” captions may become false
Colour / hue jitter	No	“red” can turn into “purple”: attribute, binding and comparison labels break
Small brightness / contrast change, light blur, noise	Yes	Objects and colours stay recognisable
Text: reorder clauses (“A to the left of B” → “to the left of B there is A”)	Yes	Same meaning; helps with the test-only phrasings
New false captions by logic (n → n±1, left → right)	Yes	Labels are certain, so it’s free extra data
Don't forget to fix your random seeds to make results reproducible.

For the individual checkpoint, design and fit proposed models on train.csv only; use validation.csv to evaluate models from Assignments 1 and 2. During the team phase, students may use the supplied labeled training and validation data for model development, including refitting on their combination. Report validation ROC-AUC only for experiments evaluated on a validation set that was held out from fitting; describe how the final model was trained.

Use test.csv only to generate predictions. It MUST NOT be used to fit preprocessing, representations, feature selection, clustering or models.

6. AI assistants
AI coding assistants and agents are allowed. However:

students remain responsible for every line of submitted code;
every computational step used to produce the submission must appear in the submitted package;
students must understand, reproduce and explain the solution;
AI assistance does not permit external data, models, segmentations or any other action with a dataset.
Chat histories are not required. Generated code is treated as the student’s own submitted work for academic-integrity purposes.

7. Intended solution and prohibited conduct
The intended task is to learn a relationship between two modalities (text and images) from the supplied labeled examples.

Any method whose primary purpose is to recover concealed information, exploit an unintended property of the dataset or platform, or obtain a score without solving the intended prediction problem is PROHIBITED, even if the method is not explicitly listed here.

The following are also prohibited:

obtaining or reconstructing hidden targets or concealed source information;
manually editing test predictions using external information;
exploiting Kaggle, Moodle, scoring, file-format or API vulnerabilities;
using multiple Kaggle accounts or circumventing submission limits;
exchanging code, engineered features, labels or predictions between teams;
making competition data or invitation links publicly available.
Ordinary feature engineering and diagnostic analysis of the supplied labeled data ARE ALLOWED within the roles defined in Section 4. If it is unclear whether a method solves the intended problem, ask the instructors before using it.

8. Kaggle submissions
Submissions are evaluated using ROC-AUC score.
The public leaderboard provides provisional feedback only.
Final ranking uses the private leaderboard after eligibility and reproducibility checks.
Each team may make at most five submissions per day.
Each team must designate one final submission.
Every submission must contain each test sample id exactly once and a numerical target probability value between 0 and 1.
The final submission must be generated by the team’s submitted code. Manual alteration of the prediction file is prohibited.

The team must record the designated Kaggle submission in its Moodle package. If Kaggle permits the selection of more than one final submission, only the submission identified in Moodle will be used for course assessment and bonuses.

9. Moodle deliverables and reproducibility
Each student submits the individual checkpoint through Moodle.

For the team phase, one designated member submits one final Moodle package on behalf of the team. The package must identify every team member and contain:

the Kaggle team name;
the team-member list and a short contribution statement;
the designated Kaggle submission identifier or filename;
one executable notebook;
any necessary source files;
the corresponding final_submission.csv;
a short experiment table containing the model architecture, validation Accuracy, runtime and seed;
package and library versions.
The notebook must:

run from a fresh kernel using Restart and Run All;
train the final model and regenerate final_submission.csv;
use relative paths and the supplied competition files;
set seeds for Python, NumPy and every model;
contain all preprocessing, feature selection/engineering, training, ensembling, calibration and prediction steps;
run without internet access or hidden local files;
complete training and prediction within 50 minutes on Kaggle GPU (Kaggle Default Hardware);
use no more VRAM than it is allowed by Kaggle Default Hardware for a final solution.
The submitted notebook and source files must not exceed 20 MB, excluding the supplied competition data and generated prediction CSV. Any external datasets or image additions excluding augmentations are prohibited.

The regenerated id values of images must match exactly as well labels from classes.csv.