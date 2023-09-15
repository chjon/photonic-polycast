#include "Common.h"
#include "Camera.cuh"
#include "Curve.cuh"
#include "Image.h"
#include "Material.cuh"
#include "Options.h"
#include "SceneNode.cuh"
#include "World.cuh"

using namespace PPCast;

uint32_t MaterialID::numMaterials = 0;

// Options
static UIntOption   opt_img_w  ("img-w"    , "the width of the image in pixels"        , 128);
static UIntOption   opt_img_h  ("img-h"    , "the height of the image in pixels"       , 128);
static StringOption opt_outfile("outfile"  , "the name of the output file"             , "img/test.png");
static UIntOption   opt_verb   ("verb"     , "verbosity (0 = none, 1 = less, 2 = more)", 2);
static UIntOption   opt_scene  ("testscene", "test scene"                              , 0);

template <class T>
static inline T& vecInsert(std::vector<T>& vec, T&& val) {
    vec.push_back(std::move(val));
    return vec.back();
}

static void makeScene(
    Camera& cam,
    std::vector<Material>& mats,
    std::vector<GeometryNode>& scene
) {
    // Default camera position and orientation
    cam.lookfrom = {0, 0, 1};
    cam.lookat   = {0, 0, 0};
    cam.up       = {0, 1, 0};

    // Set up materials
    Material normalMat     = vecInsert(mats, Material(MaterialType::NormalDir , {glm::vec3(1.0, 1.0, 1.0), 0.0f, 1.0f}));
    Material reflectionMat = vecInsert(mats, Material(MaterialType::ReflectDir, {glm::vec3(1.0, 1.0, 1.0), 0.0f, 1.0f}));
    Material refractionMat = vecInsert(mats, Material(MaterialType::RefractDir, {glm::vec3(1.0, 1.0, 1.0), 0.0f, 1.5f}));
    Material diffuseGrey   = vecInsert(mats, Material(MaterialType::Diffuse,    {glm::vec3(0.5, 0.5, 0.5), 0.0f, 1.0f}));
    Material lambertYellow = vecInsert(mats, Material(MaterialType::Lambertian, {glm::vec3(0.8, 0.8, 0.0), 0.0f, 1.0f}));
    Material lambertRed    = vecInsert(mats, Material(MaterialType::Lambertian, {glm::vec3(0.7, 0.3, 0.3), 0.0f, 1.0f}));
    Material lambertGrey   = vecInsert(mats, Material(MaterialType::Lambertian, {glm::vec3(0.5, 0.5, 0.5), 0.0f, 1.0f}));
    Material metalSilver   = vecInsert(mats, Material(MaterialType::Reflective, {glm::vec3(0.9, 0.9, 0.9), 0.0f, 1.0f}));
    Material metalFuzz     = vecInsert(mats, Material(MaterialType::Reflective, {glm::vec3(0.8, 0.6, 0.2), 0.3f, 1.0f}));
    Material glass         = vecInsert(mats, Material(MaterialType::Refractive, {glm::vec3(1.0, 1.0, 1.0), 0.0f, 1.5f}));
    Material invglass      = vecInsert(mats, Material(MaterialType::Refractive, {glm::vec3(1.0, 1.0, 1.0), 0.0f, 1.f/1.5f}));

    // Set up motion curves
    MotionCurve forward = MotionCurve{
        Curve<glm::vec3>::makeCurve<CurveType::LINEAR>({glm::vec3(0.0), glm::vec3(0.0, 0.0, 1.0)}),
        Curve<glm::mat4>::makeCurve<CurveType::LINEAR>({glm::mat4(1.0), glm::mat4(1.0)}),
        Curve<glm::vec3>::makeCurve<CurveType::LINEAR>({glm::vec3(1.0), glm::vec3(1.0)}),
    };

    // Generate scene
    switch (*opt_scene) {
        case 0:
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, normalMat)); scene.back()
                .scale(0.5);
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, refractionMat)); scene.back()
                .scale(0.5)
                .translate({-1.0, 0, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, reflectionMat)); scene.back()
                .scale(0.5)
                .translate({+1.0, 0, 0});
            break;
        case 1:
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, diffuseGrey)); scene.back()
                .scale(100)
                .translate({0, -100.5, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, diffuseGrey)); scene.back()
                .scale(0.5);
            break;
        case 2:
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, lambertGrey)); scene.back()
                .scale(100)
                .translate({0, -100.5, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, lambertGrey)); scene.back()
                .scale(0.5);
            break;
        case 3:
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, lambertYellow)); scene.back()
                .scale(100)
                .translate({0, -100.5, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, lambertRed)); scene.back()
                .scale(0.5);
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, glass)); scene.back()
                .scale(0.5)
                .translate({-1.0, 0, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, metalSilver)); scene.back()
                .scale(0.5)
                .translate({+1.0, 0, 0});
            break;
        case 4:
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, lambertYellow)); scene.back()
                .scale(100.f)
                .translate({0, -100.5, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, lambertRed)); scene.back()
                .scale(0.5f);
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, glass)); scene.back()
                .scale(0.5f)
                .translate({-1.0, 0, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, invglass)); scene.back()
                .scale(0.40f)
                .translate({-1.0, 0, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, metalFuzz)); scene.back()
                .scale(0.5f)
                .translate({+1.0, 0, 0});
            break;
        case 5:
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, glass)); scene.back()
                .scale(0.5)
                .translate({0, 0, 0.8});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, normalMat)); scene.back()
                .scale(0.5)
                .translate({0, 0, -1.0});
            break;
        case 6:
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, glass)); scene.back()
                .scale(0.5);
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, normalMat)); scene.back()
                .scale(0.5)
                .translate({0, 0, -2.0});
            break;
        case 7:
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, normalMat)); scene.back()
                .scale(0.5)
                .translate({1, 0, -3});
            scene.push_back(GeometryNode(Geometry::Primitive::Cube, metalSilver)); scene.back()
                .scale(glm::vec3(0.2f, 4.f, 4.f))
                .translate({-2.0, 0, 0})
                .rotateY(-glm::radians(45.f))
                .translate({0, 0, -3});
            scene.push_back(GeometryNode(Geometry::Primitive::Cube, metalSilver)); scene.back()
                .scale(glm::vec3(4.f, 0.2f, 4.f))
                .translate({0, -2.0, 0.0})
                .rotateY(-glm::radians(45.f))
                .translate({0, 0, -3});
            scene.push_back(GeometryNode(Geometry::Primitive::Cube, metalSilver)); scene.back()
                .scale(glm::vec3(4.f, 4.f, 0.2f))
                .translate({0, 0, -2.0})
                .rotateY(-glm::radians(45.f))
                .translate({0, 0, -3});
            break;
        case 8:
            cam.lookfrom = {-2, 2,  2};
            cam.lookat   = { 0, 0,  0};
            cam.up       = { 0, 1,  0};

            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, lambertYellow)); scene.back()
                .scale(100.f)
                .translate({0, -100.5, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, lambertRed, forward)); scene.back()
                .scale(0.5f);
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, glass)); scene.back()
                .scale(0.5f)
                .translate({-1.0, 0, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, invglass)); scene.back()
                .scale(0.40f)
                .translate({-1.0, 0, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, metalFuzz)); scene.back()
                .scale(0.5f)
                .translate({+1.0, 0, 0});
        default:
            break;
    }
}

int main(int argc, char *const *argv) {
    // Parse command line options
    if (Options::parseOptions(argc, argv)) return -1;
    if (*opt_verb >= 1) Options::printConfig(std::cout);

    // Set up camera and scene
    Camera cam;
    std::vector<Material>     materials;
    std::vector<GeometryNode> geometry;
    makeScene(cam, materials, geometry);
    World world(std::move(materials), std::move(geometry));

    // Save current camera parameters for raytracing
    cam.initialize(*opt_img_w, *opt_img_h);

    // Generate image via raytracing
    Image image(*opt_img_w, *opt_img_h);
    cam.renderImage(image, world);

    // Output image
    if (!(*opt_outfile).empty()) image.write(*opt_outfile);

    return 0;
}