close all;clear all;clc;
[I_peper,map] = imread('peppers_color.tif');
I2=I_peper(:,:,1);
figure(5);
subplot(1,2,1);
%show the image I use
imshow(I2,map);                          
I2=double(I2);
I2_=I2;I2=I2(:)
% provide - and zero
I2=I2-min(I2)+1;
%  set the histogram
s=length(I2);I2=I2(:);
remove=find(isnan(I2)==1);I2(remove)=0;
remove=find(isinf(I2)==1);I2(remove)=0;
bon=length(I2);
m=ceil(max(I2))+1;
h=zeros(1,m);
for i=1:bon,
    f=floor(I2(i));    
    if(f>0 & f<(m-1))        
        a2=I2(i)-f;
        a1=1-a2;
        h(f)  =h(f)  + a1;      
        h(f+1)=h(f+1)+ a2;                          
    end;
end;
h=conv(h,[1,2,3,2,1]);
h=h(3:(length(h)-2));h=h/sum(h);
x=find(h);h=h(x);
x=x(:);h=h(:);
% set the parameters
multi=(1:4)*m/(4+1);         
v=ones(1,4)*m;p=ones(1,4)*1/4;
difmean = mean(diff(x))/1000;
while(1)    
        P_p = dist(multi,v,p,x);
        P_sum = sum(P_p,2)+eps;
        LLH=sum(h.*log(P_sum));        
        %Max the value
        for j=1:4                          
                pp=h.*P_p(:,j)./P_sum;
                p(j) = sum(pp);
                multi(j) = sum(x.*pp)/p(j);
                vr = (x-multi(j));
                v(j)=sum(vr.*vr.*pp)/p(j)+difmean;
        end
        p = p + 1e-3;
        p = p/sum(p);
        %set break condition
        P_p = dist(multi,v,p,x);
        P_sum = sum(P_p,2)+eps;
        LLH_=sum(h.*log(P_sum));                
        if((LLH_-LLH)<0.0001) break; end;            
end            
multi=multi+min(I2)-1;  
% get the  mask
s=size(I2_);mask=zeros(s); 
for i=1:s(1),
for j=1:s(2),
  for n=1:4
    c(n)=dist(multi(n),v(n),p(n),I2_(i,j)); 
  end
  maskdata=find(c==max(c));   
  mask(i,j)=maskdata(1);
end
end
%color the image
mask(find(mask(:)==1))=45;   
mask(find(mask(:)==2))=76;
mask(find(mask(:)==3))=145;
mask(find(mask(:)==4))=76;
 subplot(1,2,2);
 imshow(ind2rgb(mask,map));
function di=dist(m,v,g,x)
x=x(:);m=m(:);
v=v(:);g=g(:);
for i=1:size(m,1)
   d = x-m(i);
   cons = g(i)/sqrt(2*pi*v(i));
   di(:,i) = cons*exp(-0.5 * (d.*d)/v(i));
end
end




