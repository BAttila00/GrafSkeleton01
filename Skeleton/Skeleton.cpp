//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
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

// vertex shader in GLSL
const char* vertexSourceForTexturing = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec2 vertexUV;			// Attrib Array 1

	out vec2 texCoord;								// output attribute

	void main() {
		texCoord = vertexUV;														// copy texture coordinates
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSourceForTexturing = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;

	in vec2 texCoord;			// variable input: interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texCoord);
	}
)";

// 2D camera
class Camera2D {
	vec2 wCenter; // középpont világkoordinátákban (világkoordinátákban megadva !!!), w kezdobetu -> világkoordinátákban
	vec2 wSize;   // szélesség és magasság világkoordinátákban (világkoordinátákban megadva !!!), w kezdobetu -> világkoordinátákban
public:
	Camera2D() : wCenter(0, 0), wSize(2, 2) { }		//amennyiben 2,2- nek adjuk meg wSize-ot, akkor az objektumaink koordinátáit(pl kör, háromszög, linestrip stb)
	//úgy kell felvenni mintha a normalizált eszközkoordináta rendszerben lennénk. Tehát ez az ablak (wSize) fog megfelelni a normalizált eszközkoordináta rendszerünknek
	//Ha pl wSize = 200,200 akkor a (80, 80)-ban felvett pont a (0.8, 0,8) koordinátákra fog leképzödni.

	mat4 V() { return TranslateMatrix(-wCenter); }		//normalizált eszközkoordináta rendszerbe való transzformáláshoz (a tanultak szerint)
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }	//normalizált eszközkoordináta rendszerbe való transzformáláshoz (a tanultak szerint)

	mat4 Vinv() { return TranslateMatrix(wCenter); }	//V transzformáció inverze
	mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }		//P transzformáció inverze

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }
	void SetBasicZoom() { wSize = vec2(2, 2); }
	void PanTo(vec2 t) { wCenter = t; }
};

Camera2D camera;		// 2D camera

GPUProgram gpuProgram; // vertex and fragment shaders
GPUProgram gpuProgramForTexturing;
//unsigned int vao;	   // virtual world on the GPU

const int nTesselatedVertices = 20;		//ennyi szakaszból rajzoljuk meg a görbéinket (pl kör)

class Circle {
	unsigned int vao, vbo;	   // virtual world on the GPU
	//int nTesselatedVertices = 20;

	float radius = 0.04;
	float circlePoints[nTesselatedVertices * 2];
	vec3 color;
	float sx, sy;		// skálázás
	vec2 wTranslate;	// eltolás
	float phi;			// forgatás szöge
public:
	Circle() {
		sx = 1.0f;
		sy = 1.0f;
		wTranslate = vec2(0.0f, 0.0f);
		phi = 0.0f;
	}

	Circle(float midPointX, float midPointY, vec3 colorParam) {
		sx = 1.0f;
		sy = 1.0f;
		wTranslate = vec2(0.0f, 0.0f);
		phi = 0.0f;
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
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point		//A vbo-ban 40 float lesz, 2 float tartozik egy csúcsponthoz így glDrawArrays()-ben csak 20 pontot kell kirajzolni (count paraméter)
			0, NULL); 		     // stride, offset: tightly packed
	}

	mat4 M() {
		mat4 Mscale(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); // scaling

		mat4 Mrotate(cosf(phi), sinf(phi), 0, 0,
			-sinf(phi), cosf(phi), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1); // rotation

		mat4 Mtranslate(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTranslate.x, wTranslate.y, 0, 1); // translation

		return Mscale * Mrotate * Mtranslate;	// model transformation
	}

	//eltolás
	void AddTranslation(vec2 wT) { wTranslate = wTranslate + wT; }

	void Draw() {
		gpuProgram.setUniform(color, "color");

		//mat4 MVPtransfasd(1, 0, 0, 0,
		//	0, 1, 0, 0,
		//	0, 0, 1, 0,
		//	0, 0, 0, 1);

		mat4 MVPtransf = M() * camera.V() * camera.P();
		gpuProgram.setUniform(MVPtransf, "MVP");

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, nTesselatedVertices /*# Elements*/);
	}


};

class LineStrip {
	unsigned int vao, vbo;	   // virtual world on the GPU
	float points[4]; //a szakaszunk kezdo és végpontja
	vec3 color;
	float sx, sy;		// skálázás
	vec2 wTranslate;	// eltolás
	float phi;			// forgatás szöge
public:

	int startNodeId, endNodeId;
	LineStrip() {
		sx = 1.0f;
		sy = 1.0f;
		wTranslate = vec2(0.0f, 0.0f);
		phi = 0.0f;
	}

	LineStrip(vec2 startPoint, vec2 endPoint, vec3 colorParam) {
		sx = 1.0f;
		sy = 1.0f;
		wTranslate = vec2(0.0f, 0.0f);
		phi = 0.0f;
		color = colorParam;
		points[0] = startPoint.x;
		points[1] = startPoint.y;
		points[2] = endPoint.x;
		points[3] = endPoint.y;

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
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point	//egy ponthoz 2 float tartozik
			0, NULL); 		     // stride, offset: tightly packed
	}

	mat4 M() {
		mat4 Mscale(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); // scaling

		mat4 Mrotate(cosf(phi), sinf(phi), 0, 0,
			-sinf(phi), cosf(phi), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1); // rotation

		mat4 Mtranslate(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTranslate.x, wTranslate.y, 0, 1); // translation

		return Mscale * Mrotate * Mtranslate;	// model transformation
	}

	//eltolás
	void AddTranslation(vec2 wT) { 
		wTranslate = wTranslate + wT;
	}

	void Draw() {
		gpuProgram.setUniform(color, "color");

		//mat4 MVPtransf(1, 0, 0, 0,
		//	0, 1, 0, 0,
		//	0, 0, 1, 0,
		//	0, 0, 0, 1);

		mat4 MVPtransf = M() * camera.V() * camera.P();
		gpuProgram.setUniform(MVPtransf, "MVP");

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, 2 /*# Elements*/);		//2 pontunk lesz amit össze kell kötni
	}

	void setPoints(vec2 startCoordinates, vec2 endCoordinates) {

		points[0] = startCoordinates.x;
		points[1] = startCoordinates.y;
		points[2] = endCoordinates.x;
		points[3] = endCoordinates.y;

		glBindVertexArray(vao);		// make it active


		//unsigned int vbo;
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(points),  // # bytes
			points,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point	//egy ponthoz 2 float tartozik
			0, NULL); 		     // stride, offset: tightly packed
	}


};

class TexturedQuad {
	unsigned int vao, vbo[2];
	vec2 vertices[4], uvs[4];
	Texture texture;
	const float size = 0.025;
	vec2 midPoint;
	float sx, sy;		// skálázás
	vec2 wTranslate;	// eltolás
	float phi;			// forgatás szöge
public:
	TexturedQuad() {
		sx = 1.0f;
		sy = 1.0f;
		wTranslate = vec2(0.0f, 0.0f);
		phi = 0.0f;
	}

	TexturedQuad(vec2 midPointParam, int width, int height, const std::vector<vec4>& image) : texture(width, height, image) {
		sx = 1.0f;
		sy = 1.0f;
		wTranslate = vec2(0.0f, 0.0f);
		phi = 0.0f;
		midPoint = midPointParam;
		vertices[0] = vec2(midPoint.x - size, midPoint.y - size); uvs[0] = vec2(0, 0);
		vertices[1] = vec2(midPoint.x + size, midPoint.y - size);  uvs[1] = vec2(1, 0);
		vertices[2] = vec2(midPoint.x + size, midPoint.y + size);   uvs[2] = vec2(1, 1);
		vertices[3] = vec2(midPoint.x - size, midPoint.y + size);  uvs[3] = vec2(0, 1);

		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		glGenBuffers(2, vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		// vertex coordinates: vbo[1] -> Attrib Array 1 -> vertexUV of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed		
	}

	mat4 M() {
		mat4 Mscale(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); // scaling

		mat4 Mrotate(cosf(phi), sinf(phi), 0, 0,
			-sinf(phi), cosf(phi), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1); // rotation

		mat4 Mtranslate(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTranslate.x, wTranslate.y, 0, 1); // translation

		return Mscale * Mrotate * Mtranslate;	// model transformation
	}

	//eltolás
	void AddTranslation(vec2 wT) { wTranslate = wTranslate + wT; }

	void Draw() {

		//mat4 MVPtransf(1, 0, 0, 0,
		//	0, 1, 0, 0,
		//	0, 0, 1, 0,
		//	0, 0, 0, 1);

		mat4 MVPtransf = M() * camera.V() * camera.P();
		gpuProgramForTexturing.setUniform(MVPtransf, "MVP");
		gpuProgramForTexturing.setUniform(texture, "textureUnit");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}

	void setMidpoint(vec2 midPointParam) {
		midPoint = midPointParam;
		vertices[0] = vec2(midPoint.x - size, midPoint.y - size);
		vertices[1] = vec2(midPoint.x + size, midPoint.y - size);
		vertices[2] = vec2(midPoint.x + size, midPoint.y + size);
		vertices[3] = vec2(midPoint.x - size, midPoint.y + size);

		glBindVertexArray(vao);		// make it active
		// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

};

class GraphNode {
	int id;
	std::vector<GraphNode> adjacentNodes;
	int width = 32, height = 32;
public:
	GraphNode() { }

	std::vector<vec4> image;
	float x, y;
	GraphNode(int idParam, float xParam, float yParam) {
		id = idParam;
		x = xParam;
		y = yParam;
		std::vector<vec4> image(width * height);
	}

	void addAdjacentNode(GraphNode node) {
		adjacentNodes.push_back(node);
	}

	bool isItConnectedToNode(int nodeNumber) {
		for (size_t i = 0; i < adjacentNodes.size(); i++)
		{
			if (nodeNumber == adjacentNodes[i].getId())
				return true;
		}
		return false;
	}

	int getId() {
		return id;
	}

	/// <summary>
	/// Kiszámolja egy adott csúc új helyét, a pont jelenlegi helye és szomszédai pozícióinak átlagolásával.
	/// </summary>
	/// <returns>A gráf csúcspontjának új koordinátái</returns>
	vec2 updatePosition() {
		float sumX = x, sumY = y;
		int numberOfAdjacentNodes = adjacentNodes.size();
		if (numberOfAdjacentNodes > 0) {
			for (size_t i = 0; i < numberOfAdjacentNodes; i++)
			{
				sumX += adjacentNodes[i].x;
				sumY += adjacentNodes[i].y;
			}
			vec2 newCoordinates = vec2(sumX / (numberOfAdjacentNodes + 1), sumY / (numberOfAdjacentNodes + 1));
			x = newCoordinates.x;
			y = newCoordinates.y;

			return newCoordinates;
		}
		else {
			return vec2(x, y);
		}
	}
};

const int numberOfVertices = 50;		//a gráfunk csúcspontjainak száma
const float fullness = 0.05f;			//a gráfunk lehetséges élei közül ennyi arányú a tényleges élek száma

//ennyi él lesz ténylegesen berajzolva
//numberOfVertices * (numberOfVertices - 1) / 2 * fullness;
const int numberOfEdges = (50 * 49 / 2) * 0.05f;

class Graph {
	GraphNode graphVertices[numberOfVertices];		// a gráfunk csúcspontjainak koordinátái
	int graphEdges[numberOfEdges * 2];						//(a gráfunk élei, minden élhez 2 koordináta) helyett minden élhez 2 csúcspontId
	std::vector<Circle> circles;
	std::vector<LineStrip> lineStrips;
	TexturedQuad* quads[numberOfVertices];
public:
	Graph() {
		//generáljuk le a csúcspontokat
		for (int i = 0; i < numberOfVertices; i++) {
			float x = generateRandomFloatBetween(-1.0f, 1.0f);
			float y = generateRandomFloatBetween(-1.0f, 1.0f);
			graphVertices[i] = GraphNode(i, x, y);
		}
		//generáljuk le az éleket
		//random kiválasztunk 2 csúcspontot a sorszámaikkal a gráf csúcsai közül, és ezek koordinátái lesznek a szakaszunk két végpontja
		for (int i = 0; i < numberOfEdges; i++) {
			int startPoint = rand() % numberOfVertices;
			int endPoint = rand() % numberOfVertices;
			//graphEdges[i * 2].x = graphVertices[startPoint].x;
			//graphEdges[i * 2].y = graphVertices[startPoint].y;
			//graphEdges[i * 2 + 1].x = graphVertices[endPoint].x;
			//graphEdges[i * 2 + 1].y = graphVertices[endPoint].y;
			graphEdges[i * 2] = startPoint;
			graphEdges[i * 2 + 1] = endPoint;
			graphVertices[startPoint].addAdjacentNode(graphVertices[endPoint]);
			graphVertices[endPoint].addAdjacentNode(graphVertices[startPoint]);
		}
	}

	void create() {
		Circle circle;
		for (int i = 0; i < numberOfVertices; i++) {
			circle = Circle(graphVertices[i].x, graphVertices[i].y, vec3(0.0f, 1.0f, 0.0f));
			circle.create();
			circles.push_back(circle);
		}

		LineStrip lineStrip;		//élek létrehozása
		for (int i = 0; i < numberOfEdges; i++) {
			int startPointId = graphEdges[2 * i];
			int endPointId = graphEdges[2 * i + 1];
			vec2 startPoint(graphVertices[startPointId].x, graphVertices[startPointId].y);
			vec2 endPoint(graphVertices[endPointId].x, graphVertices[endPointId].y);
			lineStrip = LineStrip(startPoint, endPoint, vec3(1, 1, 0));
			lineStrip.create();
			lineStrip.startNodeId = startPointId;
			lineStrip.endNodeId = endPointId;
			lineStrips.push_back(lineStrip);
		}

		//------------Textúrázott négyzeteink létrehozása---------------
		gpuProgramForTexturing.Use();
		//létre kell hoznunk egy textúrázott négyzet "mintát" -> image, ebbol fog a GPU mintavételezni, hogy a kirajzolt négyzet adott pixele milyen színü legyen
		int width = 32, height = 32;				//ez bármekkora szám lehetne (minél nagyobb annál "hirtelenebb" lesz a színátmenet)
		std::vector<vec4> image(width * height);
		for (int k = 0; k < numberOfVertices; k++) {
			//a négyzetünknek a bal és jobb oldala külön színü lesz, generáljuk le a színeket (2db)
			float leftRed = (float)(rand() % 100) / 100.0f;
			float leftGreen = (float)(rand() % 100) / 100.0f;
			float leftBlue = (float)(rand() % 100) / 100.0f;
			float rightRed = (float)(rand() % 100) / 100.0f;
			float rightGreen = (float)(rand() % 100) / 100.0f;
			float rightBlue = (float)(rand() % 100) / 100.0f;
			for (int i = 0; i < width * height; i++) {
				if (i % height < (height / 2)) {		//a négyzetünk bal oldalát ilyen színüre fessük ki
					image[i] = vec4(leftRed, leftGreen, leftBlue);
				}
				else {									//a négyzetünk jobb oldalát, pedig olyan színüre
					image[i] = vec4(rightRed, rightGreen, rightBlue);
				}
			}
			TexturedQuad* quad = new TexturedQuad(vec2(graphVertices[k].x, graphVertices[k].y), width, height, image);
			quads[k] = quad;

			graphVertices[k].image = image;
		}
		//------------!Textúrázott négyzeteink létrehozása---------------
	}

	float generateRandomFloatBetween(float from, float to) {
		return from + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (to - from)));
	}

	void AddTranslation(vec2 wT) { 
		for (int i = 0; i < numberOfVertices; i++) {
			circles[i].AddTranslation(wT);
		}
		for (int i = 0; i < numberOfEdges; i++) {
			lineStrips[i].AddTranslation(wT);
		}
		for (int i = 0; i < numberOfVertices; i++) {
			quads[i]->AddTranslation(wT);
		}
	}

	void Draw() {

		gpuProgram.Use();
		for (int i = 0; i < numberOfEdges; i++) {
			lineStrips[i].Draw();
		}

		for (int i = 0; i < numberOfVertices; i++) {
			circles[i].Draw();
		}

		gpuProgramForTexturing.Use();
		for (int k = 0; k < numberOfVertices; k++) {
			quads[k]->Draw();
		}
	}

	void Animate(float sec) {

	}

	/// <summary>
	/// kiszámolja minden csúcs új pozícióját. Ez jelen esetben a szomszédos(összekötött) csúcsok átlaga lesz
	/// </summary>
	void recalculateNodePositions() {
		for (size_t j = 0; j < 5; j++)
		{
			for (size_t i = 0; i < numberOfVertices; i++) {
				vec2 newPosition = graphVertices[i].updatePosition();
				circles[i] = Circle(newPosition.x, newPosition.y, vec3(0.0f, 1.0f, 0.0f));
				circles[i].create();

				quads[i]->setMidpoint(newPosition);
			}
			recalculateEdges();
		}
	}

	void recalculateEdges() {
		for (size_t i = 0; i < lineStrips.size(); i++)
		{
			int startNodeId = lineStrips[i].startNodeId;
			int endNodeId = lineStrips[i].endNodeId;
			vec2 startNodePosition = vec2(graphVertices[startNodeId].x, graphVertices[startNodeId].y);
			vec2 endNodePosition = vec2(graphVertices[endNodeId].x, graphVertices[endNodeId].y);
			lineStrips[i].setPoints(startNodePosition, endNodePosition);
		}
	}
};

//Circle circle =  Circle(0.5f, 0.5f, vec3(0.0f, 1.0f, 0.0f));
//Circle circle2 = Circle(0.0f, 0.0f, vec3(0.5f, 0.5f, 0.5f));
//LineStrip lineStrip;
Graph graph;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2.0f); // Width of lines in pixels

	//circle.create();
	//circle2.create();
	graph = Graph();
	graph.create();

	//lineStrip = LineStrip(0.2f, -0.5f, vec3(0, 1, 0));
	//lineStrip.create();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
	gpuProgramForTexturing.create(vertexSourceForTexturing, fragmentSourceForTexturing, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	//circle.Draw();
	//circle2.Draw();
	graph.Draw();
	//lineStrip.Draw();

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
												//ervenytelenitjuk az alkalmazoi ablakot -> ujrarajzolas

	if (key == 'a') {
		graph.AddTranslation(vec2(0.2f, 0.0f));
		glutPostRedisplay();
	}

	if (key == ' ') {
		graph.recalculateNodePositions();
		glutPostRedisplay();
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	//printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);		//kiirja az egerop rendszer szerint rzekelt pixelkoordinatak mely normavizált koordinataknak felelnek meg (akkor lesz helyes ha az ablakunk a teljes kepernyot kitolti,
															//egyebkent kicsit bonyolultabb(pontosabban, ha a viewport teljesen lefedi az alkalmazoi ablakot))

	camera.PanTo(vec2(cX, cY));
	glutPostRedisplay();
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char* buttonStat;
	switch (state) {
	case GLUT_DOWN: 
		buttonStat = "pressed"; 
		camera.Zoom(0.5f);
		glutPostRedisplay();
		break;
	case GLUT_UP:   
		buttonStat = "released"; 
		camera.SetBasicZoom();
		camera.PanTo(vec2(0, 0));
		break;
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
	float sec = time / 1000.0f;
}
