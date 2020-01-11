//
// Created by 宋孝成 on 2019-04-30.
//

#include <thread>
#include <DenseMapper.hpp>
#include <Camera.hpp>
#include <opencv2/core/persistence.hpp>
#include "System.hpp"

using namespace dtam;
using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 3)
  {
    cout << "\nUsage: executable_name path/to/data_folder_path trac_file_path\n";

    exit(0);
  }
  const std::string dataFolderPath = argv[1];std::string trac_path = argv[2];
  const std::string settingsFilePath = "/home/lzw/2_ear_DTAM/system/settings.yaml";
  System system(dataFolderPath, settingsFilePath,trac_path);

  system.start();
  //system.stop();

  return 0;
}