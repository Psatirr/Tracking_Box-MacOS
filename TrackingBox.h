#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <memory>
#include <FFGLSDK.h>
#include <ffglex/FFGLScopedFBOBinding.h>

static const int MAX_TRACKS = 8;

//Satu wajah hasil deteksi. cx, cy = pusat (0..1, kiri bawah = 0,0). size = tinggi wajah / tinggi gambar.
struct FaceRect
{
	float cx, cy, size;
};

class FaceDetector;//Didefinisikan di TrackingBox.cpp (supaya OpenCV tidak perlu di-include di sini).

//Urutan enum ini = urutan parameter yang muncul di Resolume.
enum ParamType : FFUInt32
{
	PT_QUALITY,
	PT_MODE,
	PT_TRACKS,
	PT_THRESHOLD,
	PT_TARGET_R,
	PT_TARGET_G,
	PT_TARGET_B,
	PT_TOLERANCE,
	PT_MIN_SIZE,
	PT_MAX_SIZE,
	PT_CHAOS,
	PT_SMOOTH,
	PT_SHAPE_ON,
	PT_SHAPE_STYLE,
	PT_FX_STYLE,
	PT_SHAPE_THICK,
	PT_TRACKER_R,
	PT_TRACKER_G,
	PT_TRACKER_B,
	PT_HUD_ON,
	PT_HUD_POS,
	PT_HUD_SCALE,
	PT_TEXT,
	PT_LINE_ON,
	PT_LINE_WAY,
	PT_LINE_STYLE,
	PT_CURVE,
	PT_LINE_THICK,
	PT_DASH,
	PT_LINE_R,
	PT_LINE_G,
	PT_LINE_B,
	PT_VIDEO_BG,

	PT_COUNT
};

class TrackingBox : public CFFGLPlugin
{
public:
	TrackingBox();
	~TrackingBox();

	//CFFGLPlugin
	FFResult InitGL( const FFGLViewportStruct* vp ) override;
	FFResult ProcessOpenGL( ProcessOpenGLStruct* pGL ) override;
	FFResult DeInitGL() override;

	FFResult SetFloatParameter( unsigned int dwIndex, float value ) override;
	FFResult SetTextParameter( unsigned int index, const char* value ) override;

	float GetFloatParameter( unsigned int index ) override;
	char* GetTextParameter( unsigned int index ) override;

private:
	struct Track
	{
		float x, y;      //Posisi yang digambar (sudah dihaluskan), 0..1, (0,0) = kiri bawah.
		float tx, ty;    //Posisi target dari analisis terakhir.
		float size;      //Ukuran yang digambar (0..1, hasil luas area terang).
		float tsize;     //Ukuran target.
		float alpha;     //0 = tidak terlihat, 1 = penuh. Memudar saat track hilang.
		bool detected;   //Apakah track ini ditemukan di analisis terakhir.
		int miss;        //Berapa analisis berturut-turut track ini tidak ditemukan (dipakai mode Face).
		float seed;      //Bilangan tetap per track, untuk variasi ukuran.
	};

	void AddFloatParam( unsigned int index, const char* name, unsigned int type, float defaultValue );
	void AddBoolParam( unsigned int index, const char* name, bool defaultValue );
	void AddOptionParam( unsigned int index, const char* name, int numElements, const char* const* labels, int defaultIndex );
	int OptionIndex( unsigned int index, int numElements ) const;

	struct Cand
	{
		float x, y, area;//Posisi (0..1, kiri bawah = 0,0) dan ukuran relatif calon track.
	};

	void Analyze( const unsigned char* pixels, int gw, int gh, int count, int mode );
	void MatchCandidates( const Cand* cands, int nc, int count, float aspect, int hold );
	void UpdateFaces( const FFGLTextureStruct& input, const FFGLTexCoords& maxCoords, int qualityIndex, int count );
	void ApplyFaces( const std::vector< FaceRect >& faces, int count, float aspect );
	void StepTracks( int count );
	void RebuildTextTexture();
	void BuildAtlasTexture();

	ffglex::FFGLShader analysisShader;//!< Pass 1: mengecilkan gambar jadi grid kecil berisi skor (terang / warna).
	ffglex::FFGLShader mainShader;    //!< Pass 2: menggambar gambar asli + shape + FX + garis + HUD.
	ffglex::FFGLScreenQuad quad;      //!< Quad layar penuh.
	ffglex::FFGLFBO analysisFbo;      //!< Render target kecil untuk pass 1 (ukuran maksimum, dipakai sebagian).
	std::vector< unsigned char > readback;
	std::vector< float > score;       //!< Skor tiap sel grid.
	std::vector< float > work;        //!< Salinan skor yang dipakai saat memilih puncak.
	std::vector< float > prevLum;     //!< Tingkat terang frame analisis sebelumnya (untuk mode Motion).
	bool havePrev;
	int lastGw;
	int lastGh;
	unsigned int frameCounter;
	int lastMode;

	//Mode Face (OpenCV)
	std::unique_ptr< FaceDetector > faceDetector;//!< Dibuat saat mode Face pertama kali dipakai.
	ffglex::FFGLFBO faceFbo;                     //!< Render target abu-abu kecil untuk detektor wajah.
	std::vector< unsigned char > faceGray;
	int faceFrames;
	unsigned int faceSeq;

	//Parameter
	float values[ PT_COUNT ];
	std::string text;
	bool textDirty;
	std::mutex textMutex;

	//Tekstur teks label
	GLuint textTexture;
	bool hasText;
	float textAspect;//Lebar / tinggi teks dalam pixel.
	float textUVx;   //Bagian tekstur (horizontal) yang berisi teks.
	float textUVy;   //Bagian tekstur (vertikal) yang berisi teks.

	//Tekstur atlas angka untuk HUD (karakter 0-9 : - .)
	GLuint atlasTexture;
	float atlasAspect;//Lebar sel karakter / tinggi sel. 0 = atlas tidak ada.
	float atlasCharW; //Lebar satu karakter dalam koordinat tekstur.
	float atlasUVy;   //Bagian tekstur (vertikal) yang terpakai.

	Track tracks[ MAX_TRACKS ];
	std::chrono::steady_clock::time_point startTime;
};
