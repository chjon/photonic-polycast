#include "Common.h"
#include "Camera.h"
#include "Material.h"
#include "Options.h"
#include "SceneNode.h"

using namespace PPCast;

// Options
static BoolOption   opt_usegpu ("usegpu"   , "whether to use the GPU for rendering"    , false);
static UIntOption   opt_img_w  ("img-w"    , "the width of the image in pixels"        , 128);
static UIntOption   opt_img_h  ("img-h"    , "the height of the image in pixels"       , 128);
static StringOption opt_outfile("outfile"  , "the name of the output file"             , "img/test.png");
static UIntOption   opt_verb   ("verb"     , "verbosity (0 = none, 1 = less, 2 = more)", 2);
static UIntOption   opt_scene  ("testscene", "test scene"                              , 0);
static UIntOption   opt_seed   ("seed"     , "random seed"                             , 0xDECAFBAD);

int main(int argc, char *const *argv) {
    // Parse command line options
    if (Options::parseOptions(argc, argv)) return -1;
    if (*opt_verb >= 1) Options::printConfig(std::cout);
    srand(*opt_seed);

    // Set up materials
    auto diffuseYellow = std::make_shared<MaterialDiffuse   >(glm::vec3(0.8, 0.8, 0.0));
    auto diffuseRed    = std::make_shared<MaterialDiffuse   >(glm::vec3(0.7, 0.3, 0.3));
    auto diffuseGrey   = std::make_shared<MaterialDiffuse   >(glm::vec3(0.5, 0.5, 0.5));
    auto lambertYellow = std::make_shared<MaterialLambertian>(glm::vec3(0.8, 0.8, 0.0));
    auto lambertRed    = std::make_shared<MaterialLambertian>(glm::vec3(0.7, 0.3, 0.3));
    auto lambertGrey   = std::make_shared<MaterialLambertian>(glm::vec3(0.5, 0.5, 0.5));
    auto metalSilver   = std::make_shared<MaterialMetal     >(glm::vec3(1.0, 1.0, 1.0));
    auto metalFuzz     = std::make_shared<MaterialMetal     >(glm::vec3(0.8, 0.6, 0.2), 0.3f);
    auto glass         = std::make_shared<MaterialRefractive>(glm::vec3(1.0, 1.0, 1.0), 1.1f);
    auto normalMat     = std::make_shared<MaterialNormal    >();

    // Set up camera
    const glm::vec3 pos   (0, 0, 1);
    const glm::vec3 centre(0, 0, 0);
    const glm::vec3 up    (0, 1, 0);
    const Camera cam(pos, centre, up);
    
    // Set up scene
    std::vector<GeometryNode> scene;
    switch (*opt_scene) {
        case 0:
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, normalMat)); scene.back()
                .scale(0.5);
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
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, metalSilver)); scene.back()
                .scale(0.5)
                .translate({-1.0, 0, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, metalSilver)); scene.back()
                .scale(0.5)
                .translate({+1.0, 0, 0});
            break;
        case 4:
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, diffuseYellow)); scene.back()
                .scale(100)
                .translate({0, -100.5, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, glass)); scene.back()
                .scale(0.5);
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, glass)); scene.back()
                .scale(0.5)
                .translate({-1.0, 0, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, metalSilver)); scene.back()
                .scale(0.5)
                .translate({+1.0, 0, 0});
            break;
        case 5:
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, glass)); scene.back()
                .scale(0.5)
                .translate({0, 0, 1.0});
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
        default:
            break;
    }

    // Generate image via raytracing
    png::image image = cam.render(scene, *opt_img_w, *opt_img_h);

    // Output image
    if (!(*opt_outfile).empty()) image.write(*opt_outfile);

    return 0;
}