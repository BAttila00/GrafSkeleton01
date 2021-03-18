//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
//unsigned int vao;	   // virtual world on the GPU

const int nTesselatedVertices = 20;		//ennyi szakaszb�l rajzoljuk meg a g�rb�inket (pl k�r)

class Circle {
	unsigned int vao, vbo;	   // virtual world on the GPU
	//int nTesselatedVertices = 20;

	float radius = 0.02;
	float circlePoints[nTesselatedVertices * 2];
	vec3 color;
public:
	Circle() { }

	Circle(float midPointX, float midPointY, vec3 colorParam) {
		color = colorParam;
		for (int i = 0; i < nTesselatedVertices; i++) {
			float phi = i * 2.0f * M_PI / nTesselatedVertices;
			//circlePoints.push_back(vec2(cosf(phi), sinf(phi)));
			circlePoints[2*i] = midPointX + radius * cosf(phi);
			circlePoints[2*i + 1] = midPointY + radius * sinf(phi);
		}
	}

	void create() {
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active


		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(circlePoints),  // # bytes
			circlePoints,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point		//A vbo-ban 40 float lesz, 2 float tartozik egy cs�csponthoz �gy glDrawArrays()-ben csak 20 pontot kell kirajzolni (count param�ter)
			0, NULL); 		     // stride, offset: tightly packed
	}

	void Draw() {
		gpuProgram.setUniform(color, "color");

		mat4 MVPtransf(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
		gpuProgram.setUniform(MVPtransf, "MVP");

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, nTesselatedVertices /*# Elements*/);
	}


};

class LineStrip {
	unsigned int vao, vbo;	   // virtual world on the GPU
	float points[2]; //a szakaszunk kezdo �s v�gpontja
	vec3 color;
public:
	LineStrip() { }

	LineStrip(float startPoint, float endPoint, vec3 colorParam) {
		color = colorParam;
		points[0] = startPoint;
		points[1] = endPoint;

	}

	void create() {
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active


		//unsigned int vbo;
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(points),  // # bytes
			points,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point	
			0, NULL); 		     // stride, offset: tightly packed
	}

	void Draw() {
		gpuProgram.setUniform(color, "color");

		mat4 MVPtransf(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
		gpuProgram.setUniform(MVPtransf, "MVP");

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, 2 /*# Elements*/);
	}


};

const int numberOfVertices = 50;		//a gr�funk cs�cspontjainak sz�ma
const float fullness = 0.05f;			//a gr�funk lehets�ges �lei k�z�l ennyi ar�ny� a t�nyleges �lek sz�ma

//ennyi �l lesz t�nylegesen berajzolva
//numberOfVertices * (numberOfVertices - 1) / 2 * fullness;
const int numberOfEdges = 50 * 49 / 2 * 0.05f;

class Graph {
	vec2 graphVerticesCoordinates[numberOfVertices];
	vec2 graphEdges[numberOfEdges];
public:
	Graph() {
		for (int i = 0; i < numberOfVertices; i++) {
			float x = generateRandomFloatBetween(-1.0f, 1.0f);
			float y = generateRandomFloatBetween(-1.0f, 1.0f);
			graphVerticesCoordinates[i] = vec2(x, y);
		}
	}

	float generateRandomFloatBetween(float from, float to) {
		return from + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (to - from)));
	}

	void Draw() {
		Circle circle;
		for (int i = 0; i < numberOfVertices; i++) {
			circle = Circle(graphVerticesCoordinates[i].x, graphVerticesCoordinates[i].y, vec3(0.5f, 0.5f, 0.5f));
			circle.create();
			circle.Draw();
		}
	}
};

//Circle circle =  Circle(0.5f, 0.5f, vec3(0.0f, 1.0f, 0.0f));
//Circle circle2 = Circle(0.0f, 0.0f, vec3(0.5f, 0.5f, 0.5f));
LineStrip lineStrip;
Graph graph;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(5.0f); // Width of lines in pixels

	//circle.create();
	//circle2.create();
	graph = Graph();

	lineStrip = LineStrip(0.2f, -0.5f, vec3(0, 1, 0));
	lineStrip.create();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	//circle.Draw();
	//circle2.Draw();
	graph.Draw();
	lineStrip.Draw();

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
												//ervenytelenitjuk az alkalmazoi ablakot -> ujrarajzolas
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);		//kiirja az egerop rendszer szerint rzekelt pixelkoordinatak mely normaviz�lt koordinataknak felelnek meg (akkor lesz helyes ha az ablakunk a teljes kepernyot kitolti,
															//egyebkent kicsit bonyolultabb(pontosabban, ha a viewport teljesen lefedi az alkalmazoi ablakot))
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char* buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}