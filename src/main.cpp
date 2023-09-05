#include "Common.h"
#include "Camera.h"
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
    auto lambertianRed = std::make_shared<MaterialLambertian>(glm::vec3(0.7, 0.3, 0.3));
    auto lambertianGrey = std::make_shared<MaterialLambertian>(glm::vec3(0.5, 0.5, 0.5));
    auto diffuseGreen = std::make_shared<MaterialDiffuse>(glm::vec3(0.3, 0.7, 0.3));
    auto metalWhite = std::make_shared<MaterialMetal>(glm::vec3(1.0, 1.0, 1.0));
    auto normalMat = std::make_shared<MaterialNormal>();

    // Set up camera
    const glm::vec3 pos   (0, 1, 2.5f);
    const glm::vec3 centre(0, 0, 0);
    const glm::vec3 up    (0, 1, 0);
    const Camera cam(pos, centre, up);
    
    // Set up scene
    std::vector<GeometryNode> scene;
    switch (*opt_scene) {
        case 0:
            scene.push_back(GeometryNode(Geometry::Primitive::Cube, normalMat)); scene.back()
                .scaleY(0.5)
                .rotateZ(glm::radians(45.f))
                .rotateX(glm::radians(-10.f))
                .rotateY(glm::radians(35.f));
            break;
        case 1:
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, diffuseGreen)); scene.back()
                .scale(100)
                .translate({0, -100, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, lambertianRed)); scene.back()
                .scale(0.5)
                .translate({0, +0.5, 0});
            break;
        case 2:
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, diffuseGreen)); scene.back()
                .scale(100)
                .translate({0, -100, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, metalWhite)); scene.back()
                .scale(0.5)
                .translate({0, +0.5, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, lambertianRed)); scene.back()
                .scale(0.5)
                .translate({-1.5, +0.5, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, normalMat)); scene.back()
                .scale(0.5)
                .translate({+1.5, +0.5, 0});
            break;
        case 3:
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, lambertianGrey)); scene.back()
                .scale(100)
                .translate({0, -100, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, lambertianGrey)); scene.back()
                .scale(0.5)
                .translate({0, +0.5, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, lambertianGrey)); scene.back()
                .scale(0.5)
                .translate({-1.5, +0.5, 0});
            scene.push_back(GeometryNode(Geometry::Primitive::Sphere, lambertianGrey)); scene.back()
                .scale(0.5)
                .translate({+1.5, +0.5, 0});
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