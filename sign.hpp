#include <opencv2/opencv.hpp>

using namespace std;

class Sign{
public:
    /*
    enum typ{
        ostrzegawczy =1,
        zakazu,
        nakazu,
        informacyjny
    };


    enum nazwa{
        pierwszenstwo=1,
        zakaz_wjazdu,
        stop
    };
    */
    string nazwa;
    string typ;

    cv :: Mat znak;

    int x,y;
    int w,h;

    Sign();

};
