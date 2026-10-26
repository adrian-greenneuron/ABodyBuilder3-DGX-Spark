#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdio>
#include <cstdint>
#include <locale.h>
#include <string>
#include <vector>
#include <atomic>

namespace py=pybind11;
std::atomic<uint64_t> calls{0};
template<class T>using Array=py::array_t<T,py::array::c_style|py::array::forcecast>;

template<typename... Args>void append(std::string& out,const char* format,Args... args){
    char buffer[512];
    int n=std::snprintf(buffer,sizeof(buffer),format,args...);
    if(n<0)throw std::runtime_error("PDB formatting failed");
    if(n<(int)sizeof(buffer))out.append(buffer,n);
    else{
        std::vector<char> large(n+1);
        std::snprintf(large.data(),large.size(),format,args...);
        out.append(large.data(),n);
    }
}

std::string format(Array<int64_t> aa,Array<double> pos,Array<double> mask,
                   Array<int64_t> residue,Array<double> bf,Array<int64_t> chains,
                   const std::string& remark,const std::vector<std::string>& names,
                   const std::vector<std::string>& atoms){
    const auto* a=aa.data();const auto* p=pos.data();const auto* m=mask.data();
    const auto* r=residue.data();const auto* b=bf.data();const auto* c=chains.data();
    const auto n=aa.size();const bool has_chain=chains.size()!=0;
    if(pos.size()!=n*37*3||mask.size()!=n*37||bf.size()!=n*37||residue.size()!=n||
       (has_chain&&chains.size()!=n)||names.size()!=21||atoms.size()!=37)
        throw std::invalid_argument("Unexpected PDB input dimensions");
    std::string out;out.reserve(n*37*82);
    ++calls;
    {
        py::gil_scoped_release release;
        static locale_t classic=newlocale(LC_NUMERIC_MASK,"C",nullptr);
        locale_t previous=uselocale(classic);
        struct RestoreLocale{locale_t value;~RestoreLocale(){uselocale(value);}} restore{previous};
        int64_t serial=1,previous_residue=0;char previous_chain=0;
        const char* previous_name=nullptr;
        for(py::ssize_t i=0;i<n;++i){
            int64_t type=a[i]<0?a[i]+21:a[i];
            if(type<0||type>=21)throw std::invalid_argument("Invalid residue type");
            const char* resname=names[type].c_str();
            int64_t chain=has_chain?(c[i]<0?c[i]+2:c[i]):0;
            if(has_chain&&(chain<0||chain>1))throw std::invalid_argument("Invalid chain index");
            char tag=has_chain?(chain==0?'H':'L'):'A';
            for(int atom=0;atom<37;++atom){
                auto index=i*37+atom;
                if(!(m[index]>.5))continue;
                if(serial==1){
                    if(!remark.empty())out+="REMARK "+remark+"\n";
                    out+="PARENT N/A\n";
                }
                if(previous_chain&&previous_chain!=tag){
                    append(out,"TER   %5lld      %3s %c%4lld\n",(long long)serial++,previous_name,previous_chain,(long long)previous_residue);
                    out+="PARENT N/A\n";
                }
                std::string atomname=atoms[atom].size()==4?atoms[atom]:" "+atoms[atom];
                append(out,"ATOM  %5lld %-4s %3s %c%4lld    %8.3f%8.3f%8.3f%6.2f%6.2f          %2c  \n",
                       (long long)serial++,atomname.c_str(),resname,tag,(long long)r[i],
                       p[index*3],p[index*3+1],p[index*3+2],1.0,b[index],atoms[atom][0]);
                previous_chain=tag;previous_residue=r[i];previous_name=resname;
            }
        }
        if(serial==1)return "END\n";
        append(out,"TER   %5lld      %3s %c%4lld\nEND\n",(long long)serial,previous_name,previous_chain,(long long)previous_residue);
    }
    return out;
}

PYBIND11_MODULE(_pdb_native,m){m.def("format",&format);m.def("calls",[]{return calls.load();});}
