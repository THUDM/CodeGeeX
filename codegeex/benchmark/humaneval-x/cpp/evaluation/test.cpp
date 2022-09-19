/*
Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
separate those group into separate strings and return the vector of those.
Separate groups are balanced (each open brace is properly closed) and not nested within each other
Ignore any spaces in the input string.
>>> separate_paren_groups("( ) (( )) (( )( ))")
{"()", "(())", "(()())"}
*/
#include<stdio.h>
#include<vector>
#include<string>
using namespace std;
vector<string> separate_paren_groups(string paren_string){

    vector<string> all_parens;
    string current_paren;
    int level=0;
    char chr;
    int i;
    for (i=0;i<paren_string.length();i++)
    {
        chr=paren_string[i];
        if (chr=='(')
        {
        level+=1;
        current_paren+=chr;
        }
        if (chr==')')
        {
            level-=1;
            current_paren+=chr;
            if (level==0){
                all_parens.push_back(current_paren);
                current_paren="";
            }
        }
    }
    return all_parens;
}

#undef NDEBUG
#include<assert.h>
bool issame(vector<string> a,vector<string>b){
    if (a.size()!=b.size()) return false;
    for (int i=0;i<a.size();i++)
    {
    if (a[i]!=b[i]) return false;
    }
    return true;
}
int main(){
    assert (issame(separate_paren_groups("(()()) ((())) () ((())()())"),{"(()())", "((()))", "()", "((())()())"}));
    assert (issame(separate_paren_groups("() (()) ((())) (((())))"), {"()", "(())", "((()))", "(((())))" }));
    assert (issame(separate_paren_groups("(()(())((())))") ,{ "(()(())((())))" }));
    assert (issame(separate_paren_groups("( ) (( )) (( )( ))") ,{"()", "(())", "(()())"}));
}