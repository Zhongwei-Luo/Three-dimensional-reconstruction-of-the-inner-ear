//
// Created by 宋孝成 on 2019-05-03.
//

#ifndef DTAM_MATHUTILS_HPP
#define DTAM_MATHUTILS_HPP

namespace dtam {

static inline float quadraticCorrection(float A, float B, float C) {
  return (A + C) == 2 * B ? 0.0f : (A - C) / (2 * (A - 2 * B + C));
}

}

#endif //DTAM_MATHUTILS_HPP
