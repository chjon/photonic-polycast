#include "Common.h"
#include "Camera.h"
#include "Options.h"
#include "SceneNode.h"

using namespace PPCast;

// Options
static BoolOption   opt_usegpu ("usegpu" , "whether to use the GPU for rendering"    , false);
static UIntOption   opt_img_w  ("img-w"  , "the width of the image in pixels"        , 128);
static UIntOption   opt_img_h  ("img-h"  , "the height of the image in pixels"       , 128);
static StringOption opt_outfile("outfile", "the name of the output file"             , "img/test.png");
static UIntOption   opt_verb   ("verb"   , "verbosity (0 = none, 1 = less, 2 = more)", 2);
static UIntOption   opt_env    ("testenv", "test environment"                        , 0);

int main(int argc, char *const *argv) {
    // Parse command line options
    if (Options::parseOptions(argc, argv)) return -1;
    if (*opt_verb >= 1) Options::printConfig(std::cout);

    // Set up camera
    const glm::vec3 pos   (0, 0, 2.5f);
    const glm::vec3 centre(0, 0, 0);
    const glm::vec3 up    (0, 1, 0);
    const Camera cam(pos, centre, up);

    // Set up scene
    std::vector<GeometryNode> scene;
    if (*opt_env == 0) {
        scene.push_back(GeometryNode(Geometry::Primitive::Sphere)); scene.back()
            .scaleY(0.5)
            .rotateZ(glm::radians(15.f))
            .rotateY(glm::radians(15.f));
    } else if (*opt_env == 1) {
        scene.push_back(GeometryNode(Geometry::Primitive::Cube)); scene.back()
            .rotateZ(glm::radians(45.f))
            .rotateX(glm::radians(45.f));
    }

    // Generate image via raytracing
    png::image image = cam.render(scene, *opt_img_w, *opt_img_h);

    // Output image
    if (!(*opt_outfile).empty()) image.write(*opt_outfile);

    return 0;
}