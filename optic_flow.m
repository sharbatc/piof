function [imageArray] = optic_flow(speed, theta, phi, density, direction)
% Create a 3D dot field and move a virtual camera through it.

screenid = 0;%max(Screen('Screens'));
contrast=100;
cs=contrast/100;

t_size = 51;
x_size = 151;
y_size = 91; 

%% Init OpenGL & parameters
% Trial duration in seconds
duration = 5;


directionmvt=[0 0 -1];
% In degree. Negative values orient the gaze direction toward the ground
if nargin < 1
    speed = 400;
end

if nargin < 2
    theta = 0;
end

if nargin < 3
    phi = 0;
end

rotated_direction_vector = RotateEuler(theta, phi, directionmvt);
screenWidth = 38;
dist_user=57; % cm
dot_size=0.5;
dot_color=cs*[1.0 1.0 1.0];


dimx=500;
dimy=700;
dimz=duration*speed*4;


% density=nb/Volume;
if nargin < 4 
density=0.5;
end

if nargin < 5 
direction = 1;
end

speed = direction*speed;

% Here we call some default settings for setting up Psychtoolbox
PsychDefaultSetup(2);

Screen('Preference', 'SkipSyncTests', 2);

Screen('Preference', 'VisualDebuglevel', 3);

% Initialise OpenGL
%set(gcf,'Renderer','zbuffer');

InitializeMatlabOpenGL(0,2);
[window, windowRect] = PsychImaging('OpenWindow', screenid, 0, [300 300 750 675], 32, 2, [], 2,  []);


% aspect ratio
ar = windowRect(3) / windowRect(4);
% screenWidth = screenHeight * ar;
screenHeight=screenWidth/ar;

% Number of trials
numTrials = 1;

% Maximum priority level
topPriorityLevel = MaxPriority(window);
Priority(topPriorityLevel);

    
%% creating random dot field
% Must be a 3xn matrix, n the number of 3D dots.
Volume=dimx*dimy*dimz;
% Number of dots controlled by density value
nb=round(density*10^-5*Volume);
% 3xnb matrix of coordinates
Vect=[rand(1,nb)*double(dimx)-double(dimx/2); double(300)-rand(1,nb)*double(dimy);-rand(1,nb)*double(dimz)+double(dimz/3)];


%% starting opengl context
% Start the OpenGL context (you have to do this before you issue OpenGL
% commands)
Screen('BeginOpenGL', window);

% Enable lighting
glEnable(GL.LIGHTING);

% Force there to be  ambient light  only
glLightModelfv(GL.LIGHT_MODEL_AMBIENT, [1 1 1 1]);

% Enable proper occlusion handling via depth tests
glEnable(GL.DEPTH_TEST);

% Set up a projection matrix, the projection matrix defines how images
% in our 3D simulated scene are projected to the images on our 2D monitor
glMatrixMode(GL.PROJECTION);
glLoadIdentity;
% Calculate the field of view in the y direction assuming a distance :
angle = 2 * atand((screenHeight/2) / dist_user);

% Set up our perspective projection. This is defined by our field of view
% (here given by the variable "angle") and the aspect ratio of our frustum
% (our screen) and two clipping planes. These define the minimum and
% maximum distances allowable here 0.1cm and 1000cm. If we draw outside of
% these regions then the stimuli won't be rendered
gluPerspective(angle, ar, 100, 1000);

% Setup modelview matrix: This defines the position, orientation and
% looking direction of the virtual camera that will be look at our scene.
glMatrixMode(GL.MODELVIEW);
glLoadIdentity;

% Location of the camera is at the origin
cam = [0 0 0];

% Set our camera to be looking directly down the Z axis (depth) of our
% coordinate system
fix = [0 0 -100];

% Define "up"
up = [0 1 0];

% Here we set up the attributes of our camera using the variables we have
% defined in the last three line    modelview = glGetFloatv(GL.MODELVIEW_MATRIX);
gluLookAt(cam(1), cam(2), cam(3), fix(1), fix(2), fix(3), up(1), up(2), up(3));


% Set background color to 'black' (the 'clear' color)
glClearColor(0, 0, 0, 0);

% Clear out the backbuffer
glClear;


%% List creation for the dots geometry because the geometry never changes
sphereList = glGenLists(size(Vect,2));
glNewList(sphereList, GL.COMPILE);
glMaterialfv(GL.FRONT_AND_BACK,GL.AMBIENT, dot_color);
% Draw all the dots geometry
for j=1:1:size(Vect,2)
    
    
    % Push the matrix stack
    glPushMatrix;
    
    % Translate the dots in xyz
    glTranslatef(Vect(1,j), Vect(2,j),Vect(3,j));
    
    % Draw the dots
    glutSolidSphere(dot_size, 10, 10);
    
    % Pop the matrix stack to allow creation of the next dot
    glPopMatrix;
    
end
% End of the list
glEndList;

% End the OpenGL context now that we have finished setting things up
Screen('EndOpenGL', window);

%% Start experiment
%try
% Set the frames to wait to one
waitframes = 0;
% Query the frame duration
ifi = Screen('GetFlipInterval', window);
% Sync us and get a time stamp
vbl = Screen('Flip', window);

trial = 1;

while trial<=numTrials
    % Location of the camera is at the origin
cam = [0 0 0];

% Set our camera to be looking directly down the Z axis (depth) of our
% coordinate system
fix =   cam+ rotated_direction_vector*100;

%  for motion control
last_time_stamp=GetSecs ;
% for duration control
start=last_time_stamp;
% loop
index = 0;

while ((GetSecs-start) < duration)
    index = index + 1;
    
    Screen('BeginOpenGL', window);
    glMatrixMode(GL.MODELVIEW);
    glLoadIdentity;
    
    % update of camera position in the virtual environment
    gluLookAt(cam(1), cam(2), cam(3), fix(1), fix(2), fix(3), up(1), up(2), up(3));
    
    
    % Clear out the backbuffer
    glClear;
    
    glPushMatrix;
    glCallList(sphereList);
    
    
    projection = glGetFloatv(GL.PROJECTION_MATRIX);
    modelview = glGetFloatv(GL.MODELVIEW_MATRIX);
    proj2D = reshape(projection,4,4)*reshape(modelview,4,4)*[0 0 -100 1]';
    proj2D = (proj2D./proj2D(4)+1)./2;
    glPopMatrix;
 
    
    
    % End the OpenGL context now that we have finished
    Screen('EndOpenGL', window);
    
    % Flip to the screen
    % Show rendered image at next vertical retrace
    vbl = Screen('Flip', window, vbl + (waitframes - 0.5) * ifi);
    % update cam position for next frame update according to the time between frames and speed ( speed)
    cam=cam + directionmvt*double(speed)*double(GetSecs-last_time_stamp);
    last_time_stamp=GetSecs;
    % update fix coordinate according to the current cam position
    fix=cam+ rotated_direction_vector;
    
    

    
    imageArray(:,:,index) = rgb2gray(Screen('GetImage', window)); %,nrchannels=3])
    %cam_position(:,:,index) = campos;
end
    trial = trial + 1;
    disp(trial);  
end

Screen('CloseAll')

    imageArray = imageArray(:, :, 1:t_size);
    imageArray = imresize(imageArray, [y_size, x_size], 'bilinear');
    imageArray(imageArray > 0) = 255;
    proj2D_FoE = proj2D.*[151; 91; 1; 1;];
    proj2D_FoE(1:2)
end
